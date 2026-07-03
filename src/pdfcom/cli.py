from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence

from .optimizer import OptimizationResult, optimize_pdf, plan_label


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
    inputs: tuple[Path, ...]


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    clobber = bool(namespace.clobber and not namespace.no_clobber)

    if not namespace.dry_run and not namespace.inplace and namespace.output_dir is None:
        parser.error("one of -i or -o is required unless -n is used")

    options = CliOptions(
        dry_run=namespace.dry_run,
        recurse=namespace.recurse,
        inplace=namespace.inplace,
        output_dir=namespace.output_dir,
        clobber=clobber,
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
    for record in records:
        ok = process_one(record, options)
        had_error = had_error or not ok

    return 1 if had_error else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdfcom",
        description="Try multiple PDF compression pipelines and keep the smallest valid result.",
    )
    parser.add_argument("-n", dest="dry_run", action="store_true", help="dry run")
    parser.add_argument("-r", dest="recurse", action="store_true", help="recurse into directories")
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


def process_one(record: InputPdf, options: CliOptions) -> bool:
    target = options.output_dir / record.output_rel if options.output_dir is not None else None
    if target is not None and target.exists() and not options.clobber and not options.dry_run:
        print(f"{record.display}: skip; output exists: {target}", file=sys.stderr)
        return False

    try:
        with optimize_pdf(record.source) as result:
            print(summary_line(record.display, result))
            if options.dry_run:
                print(dry_run_action(record, options, result))
                return True
            if options.inplace:
                return write_inplace(record, result)
            if target is None:
                print(f"{record.display}: no output mode selected", file=sys.stderr)
                return False
            return write_to_output(record, result, target, options.clobber)
    except Exception as exc:  # noqa: BLE001 - report file-level failure and continue.
        print(f"{record.display}: error: {exc}", file=sys.stderr)
        return False


def summary_line(display: str, result: OptimizationResult) -> str:
    plan = plan_label(result.best.plan)
    saved = max(result.saved_bytes, 0)
    return (
        f"{display}: {format_bytes(result.original_size)} -> {format_bytes(result.best.size)}; "
        f"save {format_bytes(saved)} ({max(result.saved_percent, 0):.2f}%) via {plan}"
    )


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


def write_inplace(record: InputPdf, result: OptimizationResult) -> bool:
    if not result.reduced:
        print(f"{record.display}: left unchanged")
        return True
    atomic_copy(result.best.path, record.source, mode_source=record.source, preserve_metadata=False)
    print(f"{record.display}: replaced in place")
    return True


def write_to_output(
    record: InputPdf,
    result: OptimizationResult,
    target: Path,
    clobber: bool,
) -> bool:
    if target.exists() and not clobber:
        print(f"{record.display}: skip; output exists: {target}", file=sys.stderr)
        return False

    content_source = result.best.path if result.reduced else record.source
    preserve_metadata = not result.reduced
    atomic_copy(content_source, target, mode_source=record.source, preserve_metadata=preserve_metadata)
    action = "wrote optimized output" if result.reduced else "copied original output"
    print(f"{record.display}: {action}: {target}")
    return True


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
