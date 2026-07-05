from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations, permutations, product
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Iterator
import warnings

import pikepdf
import pymupdf


class Method(StrEnum):
    GHOSTSCRIPT = "ghostscript"
    PIKEPDF = "pikepdf"
    PYMUPDF = "pymupdf"
    QPDF = "qpdf"


ALL_METHODS: tuple[Method, ...] = (
    Method.GHOSTSCRIPT,
    Method.PIKEPDF,
    Method.PYMUPDF,
    Method.QPDF,
)

DEFAULT_PLAN_SPEC = "gs?,{mu?,pike?},q"
FULL_PERMUTATION_PLAN_SPEC = "{gs?,pike?,mu?,q?}"
METHOD_NAMES = {
    "gs": Method.GHOSTSCRIPT,
    "pike": Method.PIKEPDF,
    "mu": Method.PYMUPDF,
    "q": Method.QPDF,
}


@dataclass(frozen=True)
class Candidate:
    plan: tuple[Method, ...]
    path: Path
    size: int


@dataclass(frozen=True)
class PlanResult:
    plan: tuple[Method, ...]
    ok: bool
    size: int | None = None
    page_count: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class PlanSelection:
    spec: str
    plans: tuple[tuple[Method, ...], ...]
    unavailable_methods: tuple[Method, ...]


@dataclass(frozen=True)
class OptimizationResult:
    source: Path
    original_size: int
    original_page_count: int
    best: Candidate
    attempted_plans: int
    failed_plans: dict[str, str]
    unavailable_methods: tuple[Method, ...]
    plan_results: tuple[PlanResult, ...]

    @property
    def reduced(self) -> bool:
        return self.best.size < self.original_size

    @property
    def saved_bytes(self) -> int:
        return self.original_size - self.best.size

    @property
    def saved_percent(self) -> float:
        if self.original_size == 0:
            return 0.0
        return self.saved_bytes / self.original_size * 100


def plan_label(plan: tuple[Method, ...]) -> str:
    if not plan:
        return "original"
    return "+".join(method.value for method in plan)


def select_plans(spec: str = DEFAULT_PLAN_SPEC) -> PlanSelection:
    requested_plans = expand_plan_spec(spec)
    requested_methods = tuple(dict.fromkeys(method for plan in requested_plans for method in plan))
    unavailable = unavailable_methods(requested_methods)
    plans = dedupe_plans(
        tuple(method for method in plan if method not in unavailable)
        for plan in requested_plans
    )
    return PlanSelection(spec=spec, plans=plans, unavailable_methods=unavailable)


def select_full_permutation_plans() -> PlanSelection:
    unavailable = unavailable_methods(ALL_METHODS)
    plans = dedupe_plans(
        tuple(method for method in plan if method not in unavailable)
        for plan in optimization_plans(ALL_METHODS)
    )
    return PlanSelection(
        spec=FULL_PERMUTATION_PLAN_SPEC,
        plans=plans,
        unavailable_methods=unavailable,
    )


def unavailable_methods(methods: tuple[Method, ...]) -> tuple[Method, ...]:
    unavailable = []
    for method in methods:
        if method is Method.GHOSTSCRIPT and shutil.which("gs") is None:
            unavailable.append(method)
        elif method is Method.QPDF and shutil.which("qpdf") is None:
            unavailable.append(method)
    return tuple(unavailable)


def optimization_plans(methods: tuple[Method, ...]) -> list[tuple[Method, ...]]:
    plans: list[tuple[Method, ...]] = []
    for length in range(1, len(methods) + 1):
        plans.extend(permutations(methods, length))
    return plans


def expand_plan_spec(spec: str) -> tuple[tuple[Method, ...], ...]:
    terms = split_top_level(spec)
    if not terms:
        raise ValueError("empty plan spec")

    term_choices = [expand_plan_term(term) for term in terms]
    plans = []
    for choices in product(*term_choices):
        plan = tuple(method for choice in choices for method in choice)
        plans.append(plan)
    return dedupe_plans(plans)


def expand_plan_term(term: str) -> tuple[tuple[Method, ...], ...]:
    term = term.strip()
    if not term:
        raise ValueError("empty plan term")

    group_optional = term.endswith("?")
    group = term[:-1].strip() if group_optional else term
    if group.startswith("{") or group.startswith("("):
        opener = group[0]
        closer = "}" if opener == "{" else ")"
        if not group.endswith(closer):
            raise ValueError(f"unterminated plan group: {term}")
        inner = group[1:-1]
        items = parse_plan_items(inner)
        choices = expand_plan_items(items, permute=opener == "{")
        if group_optional:
            choices = (*choices, ())
        return dedupe_plan_choices(choices)

    if term.startswith("}") or term.startswith(")") or term.startswith("{") or term.startswith("("):
        raise ValueError(f"invalid plan term: {term}")

    method, optional = parse_plan_item(term)
    choices: tuple[tuple[Method, ...], ...] = ((method,),)
    if optional:
        choices = (*choices, ())
    return choices


def parse_plan_items(inner: str) -> tuple[tuple[Method, bool], ...]:
    terms = split_top_level(inner)
    if not terms:
        raise ValueError("empty plan group")
    items = tuple(parse_plan_item(term) for term in terms)
    methods = [method for method, _optional in items]
    if len(set(methods)) != len(methods):
        raise ValueError("duplicate method in plan group")
    return items


def parse_plan_item(item: str) -> tuple[Method, bool]:
    item = item.strip()
    optional = item.endswith("?")
    name = item[:-1].strip() if optional else item
    if not name:
        raise ValueError("empty method name in plan spec")
    try:
        return METHOD_NAMES[name], optional
    except KeyError as exc:
        raise ValueError(f"unknown method in plan spec: {name}") from exc


def expand_plan_items(
    items: tuple[tuple[Method, bool], ...],
    *,
    permute: bool,
) -> tuple[tuple[Method, ...], ...]:
    optional_methods = [method for method, optional in items if optional]
    choices = []
    for included_optional in optional_subsets(optional_methods):
        selected = tuple(
            method
            for method, optional in items
            if not optional or method in included_optional
        )
        if not selected:
            choices.append(())
        elif permute:
            choices.extend(permutations(selected))
        else:
            choices.append(selected)
    return dedupe_plan_choices(choices)


def optional_subsets(methods: list[Method]) -> tuple[frozenset[Method], ...]:
    subsets = []
    count = len(methods)
    for length in range(count, -1, -1):
        subsets.extend(frozenset(subset) for subset in combinations(methods, length))
    return tuple(subsets)


def split_top_level(value: str) -> list[str]:
    terms = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "{(":
            depth += 1
        elif char in "})":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unmatched closing delimiter in plan spec: {value}")
        elif char == "," and depth == 0:
            terms.append(value[start:index].strip())
            start = index + 1
    if depth != 0:
        raise ValueError(f"unmatched opening delimiter in plan spec: {value}")
    final = value[start:].strip()
    if final:
        terms.append(final)
    elif value.strip().endswith(","):
        raise ValueError("plan spec cannot end with a comma")
    return terms


def dedupe_plan_choices(plans: Iterable[tuple[Method, ...]]) -> tuple[tuple[Method, ...], ...]:
    deduped = []
    seen = set()
    for plan in plans:
        if plan in seen:
            continue
        seen.add(plan)
        deduped.append(plan)
    return tuple(deduped)


def dedupe_plans(plans: Iterable[tuple[Method, ...]]) -> tuple[tuple[Method, ...], ...]:
    return tuple(plan for plan in dedupe_plan_choices(plans) if plan)


@contextmanager
def optimize_pdf(
    source: Path,
    *,
    plans: tuple[tuple[Method, ...], ...] | None = None,
    unavailable_methods: tuple[Method, ...] = (),
) -> Iterator[OptimizationResult]:
    with tempfile.TemporaryDirectory(prefix="pdfcom-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        if plans is None:
            selection = select_plans()
            plans = selection.plans
            unavailable_methods = selection.unavailable_methods
        yield _optimize_pdf(
            source,
            temp_dir,
            plans=plans,
            unavailable_methods=unavailable_methods,
        )


def _optimize_pdf(
    source: Path,
    temp_dir: Path,
    *,
    plans: tuple[tuple[Method, ...], ...] | None = None,
    unavailable_methods: tuple[Method, ...] = (),
) -> OptimizationResult:
    source = source.resolve()
    original_size = source.stat().st_size
    original_page_count = page_count(source)
    best = Candidate((), source, original_size)
    if plans is None:
        selection = select_plans()
        plans = selection.plans
        unavailable_methods = selection.unavailable_methods
    failures: dict[str, str] = {}
    plan_results: list[PlanResult] = []
    attempted = 0

    for plan_index, plan in enumerate(plans, start=1):
        attempted += 1
        current = source
        try:
            for stage_index, method in enumerate(plan, start=1):
                output = temp_dir / f"{plan_index:02d}-{stage_index:02d}-{method.value}.pdf"
                run_method(method, current, output)
                if not output.is_file():
                    raise RuntimeError(f"{method.value} did not create an output file")
                current = output

            candidate_size = current.stat().st_size
            candidate_page_count = page_count(current)
            if candidate_page_count != original_page_count:
                raise RuntimeError(
                    "page count changed "
                    f"from {original_page_count} to {candidate_page_count}"
                )
            plan_results.append(
                PlanResult(
                    plan=plan,
                    ok=True,
                    size=candidate_size,
                    page_count=candidate_page_count,
                )
            )
            if candidate_size < best.size:
                best = Candidate(plan, current, candidate_size)
        except Exception as exc:  # noqa: BLE001 - keep trying other compressors.
            failures[plan_label(plan)] = str(exc)
            plan_results.append(PlanResult(plan=plan, ok=False, error=str(exc)))

    return OptimizationResult(
        source=source,
        original_size=original_size,
        original_page_count=original_page_count,
        best=best,
        attempted_plans=attempted,
        failed_plans=failures,
        unavailable_methods=unavailable_methods,
        plan_results=tuple(plan_results),
    )


def page_count(path: Path) -> int:
    try:
        with pikepdf.Pdf.open(path) as pdf:
            return len(pdf.pages)
    except Exception:
        with pymupdf.open(path) as document:
            return document.page_count


def run_method(method: Method, source: Path, output: Path) -> None:
    if method is Method.GHOSTSCRIPT:
        run_ghostscript(source, output)
    elif method is Method.PIKEPDF:
        run_pikepdf(source, output)
    elif method is Method.PYMUPDF:
        run_pymupdf(source, output)
    elif method is Method.QPDF:
        run_qpdf(source, output)
    else:
        raise ValueError(f"unknown optimizer method: {method}")


def run_ghostscript(source: Path, output: Path) -> None:
    gs = shutil.which("gs")
    if gs is None:
        raise RuntimeError("ghostscript executable not found: gs")

    command = [
        gs,
        "-sDEVICE=pdfwrite",
        "-dNOPAUSE",
        "-dBATCH",
        "-dSAFER",
        "-dCompatibilityLevel=1.7",
        "-dPDFSETTINGS=/prepress",
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dSubsetFonts=true",
        "-dCompressPages=true",
        "-dAutoRotatePages=/None",
        "-dDownsampleColorImages=false",
        "-dDownsampleGrayImages=false",
        "-dDownsampleMonoImages=false",
        f"-sOutputFile={output}",
        str(source),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if len(stderr) > 500:
            stderr = stderr[-500:]
        raise RuntimeError(f"ghostscript failed with exit {completed.returncode}: {stderr}")


def run_pikepdf(source: Path, output: Path) -> None:
    with pikepdf.Pdf.open(source) as pdf:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=pikepdf.PageCopyWarning)
            pdf.save(
                output,
                compress_streams=True,
                recompress_flate=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
                linearize=False,
                deterministic_id=True,
            )


def run_pymupdf(source: Path, output: Path) -> None:
    with pymupdf.open(source) as document:
        if document.needs_pass:
            raise RuntimeError("encrypted PDF requires a password")
        document.save(output, garbage=4, deflate=True)


def run_qpdf(source: Path, output: Path) -> None:
    qpdf = shutil.which("qpdf")
    if qpdf is None:
        raise RuntimeError("qpdf executable not found: qpdf")

    command = [
        qpdf,
        "--compress-streams=y",
        "--decode-level=generalized",
        "--recompress-flate",
        "--compression-level=9",
        "--object-streams=generate",
        "--deterministic-id",
        str(source),
        str(output),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if len(stderr) > 500:
            stderr = stderr[-500:]
        raise RuntimeError(f"qpdf failed with exit {completed.returncode}: {stderr}")
