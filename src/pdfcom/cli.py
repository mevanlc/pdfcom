from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable, Sequence

from .optimizer import (
    DEFAULT_PLAN,
    Method,
    OptimizationResult,
    optimize_pdf,
    parse_verification_spec,
    plan_label,
    select_plan_method_permutations,
    select_plans,
    VerificationCheck,
    VerificationResult,
)


@dataclass(frozen=True)
class InputPdf:
    source: Path
    display: str
    output_rel: Path


@dataclass(frozen=True)
class CliOptions:
    dry_run: bool
    recurse: bool
    inplace: bool
    output_dir: Path | None
    clobber: bool
    workers: int
    plan_spec: str
    plans: tuple[tuple[Method, ...], ...]
    unavailable_methods: tuple[Method, ...]
    test_spec: str | None
    verification_checks: tuple[VerificationCheck, ...]
    stats_file: Path | None
    inputs: tuple[Path, ...]


@dataclass(frozen=True)
class ProcessReport:
    ok: bool
    stdout: tuple[str, ...] = ()
    stderr: tuple[str, ...] = ()
    stats: dict[str, object] | None = None


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    clobber = bool(namespace.clobber and not namespace.no_clobber)

    if not namespace.dry_run and not namespace.inplace and namespace.output_dir is None:
        parser.error("one of -i or -o is required unless -n is used")

    try:
        plan_spec = namespace.plan_spec or DEFAULT_PLAN
        if namespace.full_permutation:
            plan_selection = select_plan_method_permutations(plan_spec)
        else:
            plan_selection = select_plans(plan_spec)
        verification_checks = parse_verification_spec(namespace.test_spec)
    except ValueError as exc:
        parser.error(str(exc))

    options = CliOptions(
        dry_run=namespace.dry_run,
        recurse=namespace.recurse,
        inplace=namespace.inplace,
        output_dir=namespace.output_dir,
        clobber=clobber,
        workers=namespace.workers,
        plan_spec=plan_selection.spec,
        plans=plan_selection.plans,
        unavailable_methods=plan_selection.unavailable_methods,
        test_spec=namespace.test_spec,
        verification_checks=verification_checks,
        stats_file=namespace.stats_file,
        inputs=tuple(namespace.inputs),
    )

    records, errors = discover_inputs(options.inputs, options.recurse)
    for error in errors:
        print(f"pdfcom: {error}", file=sys.stderr)

    if options.output_dir is not None:
        collision_errors = output_collision_errors(records, options.output_dir)
        for error in collision_errors:
            print(f"pdfcom: {error}", file=sys.stderr)
        if collision_errors:
            return 1

    if not records:
        print("pdfcom: no PDF files matched", file=sys.stderr)
        return 1

    if options.output_dir is not None and not options.dry_run:
        options.output_dir.mkdir(parents=True, exist_ok=True)
        if not options.output_dir.is_dir():
            print(f"pdfcom: output path is not a directory: {options.output_dir}", file=sys.stderr)
            return 1

    had_error = bool(errors)
    stats_records: list[dict[str, object]] = []
    for report in process_records(records, options):
        emit_report(report)
        had_error = had_error or not report.ok
        if report.stats is not None:
            stats_records.append(report.stats)

    if options.stats_file is not None:
        write_stats_file(options.stats_file, options, records, stats_records)

    return 1 if had_error else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdfcom",
        description="Try multiple PDF compression pipelines and keep the smallest valid result.",
    )
    parser.add_argument("-n", dest="dry_run", action="store_true", help="dry run")
    parser.add_argument(
        "-p",
        dest="plan_spec",
        metavar="SPEC",
        help=f"compression plan spec; default: {DEFAULT_PLAN}",
    )
    parser.add_argument(
        "-P",
        dest="full_permutation",
        action="store_true",
        help="try all non-empty permutations of methods in the plan spec",
    )
    parser.add_argument("-r", dest="recurse", action="store_true", help="recurse into directories")
    parser.add_argument(
        "-t",
        dest="test_spec",
        metavar="SPEC",
        help="post-compression verification test spec",
    )
    parser.add_argument(
        "-s",
        dest="stats_file",
        metavar="FILE",
        type=Path,
        help="write per-plan optimization stats as JSON",
    )
    parser.add_argument(
        "-w",
        dest="workers",
        metavar="N",
        type=positive_int,
        default=1,
        help="number of worker processes",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-i", dest="inplace", action="store_true", help="replace reducible PDFs in place")
    mode.add_argument("-o", dest="output_dir", metavar="DIR", type=Path, help="write outputs to DIR")
    parser.add_argument("-c", dest="clobber", action="store_true", help="clobber existing outputs")
    parser.add_argument(
        "-C",
        dest="no_clobber",
        action="store_true",
        help="do not clobber existing outputs; overrides -c",
    )
    parser.add_argument("inputs", metavar="pdf_file_or_dir", nargs="+", type=Path)
    return parser


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def discover_inputs(inputs: tuple[Path, ...], recurse: bool) -> tuple[list[InputPdf], list[str]]:
    records: list[InputPdf] = []
    errors: list[str] = []
    seen_sources: set[Path] = set()

    for input_path in inputs:
        if input_path.is_file():
            if not is_pdf(input_path):
                errors.append(f"not a PDF file: {input_path}")
                continue
            add_record(
                records,
                errors,
                seen_sources,
                input_path,
                input_path.as_posix(),
                Path(input_path.name),
            )
            continue

        if input_path.is_dir():
            records_for_dir = discover_dir(input_path, recurse)
            for source, output_rel in records_for_dir:
                add_record(
                    records,
                    errors,
                    seen_sources,
                    source,
                    source.as_posix(),
                    output_rel,
                )
            continue

        errors.append(f"input does not exist: {input_path}")

    return records, errors


def discover_dir(directory: Path, recurse: bool) -> list[tuple[Path, Path]]:
    if recurse:
        candidates = sorted(path for path in directory.rglob("*") if path.is_file() and is_pdf(path))
    else:
        candidates = sorted(path for path in directory.iterdir() if path.is_file() and is_pdf(path))

    prefix = directory.name or directory.resolve().name
    records = []
    for source in candidates:
        rel_inside = source.relative_to(directory)
        records.append((source, Path(prefix) / rel_inside))
    return records


def add_record(
    records: list[InputPdf],
    errors: list[str],
    seen_sources: set[Path],
    source: Path,
    display: str,
    output_rel: Path,
) -> None:
    resolved = source.resolve()
    if resolved in seen_sources:
        errors.append(f"duplicate input skipped: {source}")
        return
    seen_sources.add(resolved)
    records.append(InputPdf(source=source, display=display, output_rel=output_rel))


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def output_collision_errors(records: list[InputPdf], output_dir: Path) -> list[str]:
    errors = []
    seen: dict[Path, Path] = {}
    for record in records:
        target = (output_dir / record.output_rel).resolve()
        source = record.source.resolve()
        if target == source:
            errors.append(f"output path matches input; use -i instead: {record.source}")
            continue
        other = seen.get(target)
        if other is not None:
            errors.append(f"multiple inputs map to output {target}: {other} and {record.source}")
            continue
        seen[target] = record.source
    return errors


def process_records(records: list[InputPdf], options: CliOptions) -> Iterable[ProcessReport]:
    if options.workers == 1 or len(records) <= 1:
        for record in records:
            yield process_one(record, options)
        return

    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=options.workers) as pool:
        yield from pool.imap_unordered(process_one_job, ((record, options) for record in records))


def process_one_job(job: tuple[InputPdf, CliOptions]) -> ProcessReport:
    record, options = job
    return process_one(record, options)


def emit_report(report: ProcessReport) -> None:
    for line in report.stdout:
        print(line)
    for line in report.stderr:
        print(line, file=sys.stderr)


def process_one(record: InputPdf, options: CliOptions) -> ProcessReport:
    stdout: list[str] = []
    stderr: list[str] = []
    target = options.output_dir / record.output_rel if options.output_dir is not None else None
    if target is not None and target.exists() and not options.clobber and not options.dry_run:
        stderr.append(f"{record.display}: skip; output exists: {target}")
        return ProcessReport(False, stderr=tuple(stderr))

    try:
        with optimize_pdf(
            record.source,
            plans=options.plans,
            unavailable_methods=options.unavailable_methods,
            verification_checks=options.verification_checks,
        ) as result:
            stdout.append(summary_line(record.display, result))
            stderr.extend(result_warnings(record.display, result))
            if options.dry_run:
                stdout.append(dry_run_action(record, options, result))
                return ProcessReport(
                    True,
                    stdout=tuple(stdout),
                    stderr=tuple(stderr),
                    stats=optional_stats(record, result, options),
                )
            if options.inplace:
                ok, message = write_inplace(record, result)
                stdout.append(message)
                return ProcessReport(
                    ok,
                    stdout=tuple(stdout),
                    stderr=tuple(stderr),
                    stats=optional_stats(record, result, options),
                )
            if target is None:
                stderr.append(f"{record.display}: no output mode selected")
                return ProcessReport(
                    False,
                    stdout=tuple(stdout),
                    stderr=tuple(stderr),
                    stats=optional_stats(record, result, options),
                )
            ok, message, error = write_to_output(record, result, target, options.clobber)
            if message is not None:
                stdout.append(message)
            if error is not None:
                stderr.append(error)
            return ProcessReport(
                ok,
                stdout=tuple(stdout),
                stderr=tuple(stderr),
                stats=optional_stats(record, result, options),
            )
    except Exception as exc:  # noqa: BLE001 - report file-level failure and continue.
        stderr.append(f"{record.display}: error: {exc}")
        return ProcessReport(False, stdout=tuple(stdout), stderr=tuple(stderr))


def summary_line(display: str, result: OptimizationResult) -> str:
    plan = plan_label(result.best.plan)
    saved = max(result.saved_bytes, 0)
    return (
        f"{display}: {format_bytes(result.original_size)} -> {format_bytes(result.best.size)}; "
        f"save {format_bytes(saved)} ({max(result.saved_percent, 0):.2f}%) via {plan}"
    )


def stats_record(record: InputPdf, result: OptimizationResult) -> dict[str, object]:
    return {
        "source": record.source.as_posix(),
        "display": record.display,
        "output_rel": record.output_rel.as_posix(),
        "original_size": result.original_size,
        "original_page_count": result.original_page_count,
        "source_error": result.source_error,
        "best_plan": plan_label(result.best.plan),
        "best_methods": [method.value for method in result.best.plan],
        "best_size": result.best.size,
        "saved_bytes": result.saved_bytes,
        "saved_percent": result.saved_percent,
        "reduced": result.reduced,
        "attempted_plans": result.attempted_plans,
        "successful_plans": result.successful_plans,
        "all_plans_failed": result.all_plans_failed,
        "unavailable_methods": [method.value for method in result.unavailable_methods],
        "plans": [
            {
                "plan": plan_label(plan_result.plan),
                "methods": [method.value for method in plan_result.plan],
                "ok": plan_result.ok,
                "size": plan_result.size,
                "page_count": plan_result.page_count,
                "saved_bytes": (
                    result.original_size - plan_result.size if plan_result.size is not None else None
                ),
                "saved_percent": (
                    (result.original_size - plan_result.size) / result.original_size * 100
                    if plan_result.size is not None and result.original_size
                    else None
                ),
                "error": plan_result.error,
                "verification": [
                    verification_stats_record(verification_result)
                    for verification_result in plan_result.verification_results
                ],
            }
            for plan_result in result.plan_results
        ],
        "failed_plans": result.failed_plans,
    }


def optional_stats(
    record: InputPdf,
    result: OptimizationResult,
    options: CliOptions,
) -> dict[str, object] | None:
    if options.stats_file is None:
        return None
    return stats_record(record, result)


def write_stats_file(
    path: Path,
    options: CliOptions,
    records: list[InputPdf],
    stats_records: list[dict[str, object]],
) -> None:
    payload = {
        "version": 1,
        "options": {
            "dry_run": options.dry_run,
            "recurse": options.recurse,
            "inplace": options.inplace,
            "output_dir": options.output_dir.as_posix() if options.output_dir is not None else None,
            "clobber": options.clobber,
            "workers": options.workers,
            "plan_spec": options.plan_spec,
            "plans": [[method.value for method in plan] for plan in options.plans],
            "unavailable_methods": [method.value for method in options.unavailable_methods],
            "test_spec": options.test_spec,
            "verification": [verification_check_stats_record(check) for check in options.verification_checks],
            "inputs": [path.as_posix() for path in options.inputs],
        },
        "input_count": len(records),
        "results": sorted(stats_records, key=lambda item: str(item["display"])),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def result_warnings(display: str, result: OptimizationResult) -> tuple[str, ...]:
    if result.source_failed:
        return (
            f"{display}: warning: input could not be opened as PDF; "
            f"treating as incompressible: {result.source_error}",
        )
    if result.all_plans_failed:
        return (f"{display}: warning: all optimization plans failed; treating as incompressible",)
    return ()


def verification_check_stats_record(check: VerificationCheck) -> dict[str, object]:
    return {
        "tool": check.tool.value,
        "test": check.test.value,
        "slack": check.slack,
        "strip_ws": check.strip_ws,
    }


def verification_stats_record(result: VerificationResult) -> dict[str, object]:
    return {
        **verification_check_stats_record(result.check),
        "ok": result.ok,
        "original_count": result.original_count,
        "candidate_count": result.candidate_count,
        "delta": result.delta,
        "allowed_delta": result.allowed_delta,
        "error": result.error,
    }


def dry_run_action(record: InputPdf, options: CliOptions, result: OptimizationResult) -> str:
    if options.inplace:
        if result.reduced:
            return f"{record.display}: would replace in place"
        return f"{record.display}: would leave original unchanged"

    if options.output_dir is not None:
        target = options.output_dir / record.output_rel
        if target.exists() and not options.clobber:
            return f"{record.display}: would skip; output exists: {target}"
        if result.reduced:
            return f"{record.display}: would write optimized output: {target}"
        return f"{record.display}: would copy original output: {target}"

    return f"{record.display}: dry run only; no output mode selected"


def write_inplace(record: InputPdf, result: OptimizationResult) -> tuple[bool, str]:
    if not result.reduced:
        return True, f"{record.display}: left unchanged"
    atomic_copy(result.best.path, record.source, mode_source=record.source, preserve_metadata=False)
    return True, f"{record.display}: replaced in place"


def write_to_output(
    record: InputPdf,
    result: OptimizationResult,
    target: Path,
    clobber: bool,
) -> tuple[bool, str | None, str | None]:
    if target.exists() and not clobber:
        return False, None, f"{record.display}: skip; output exists: {target}"

    content_source = result.best.path if result.reduced else record.source
    preserve_metadata = not result.reduced
    atomic_copy(content_source, target, mode_source=record.source, preserve_metadata=preserve_metadata)
    action = "wrote optimized output" if result.reduced else "copied original output"
    return True, f"{record.display}: {action}: {target}", None


def atomic_copy(
    content_source: Path,
    target: Path,
    *,
    mode_source: Path,
    preserve_metadata: bool,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        if preserve_metadata:
            shutil.copy2(content_source, temp_path)
        else:
            shutil.copyfile(content_source, temp_path)
            shutil.copymode(mode_source, temp_path)
        os.replace(temp_path, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def format_bytes(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
