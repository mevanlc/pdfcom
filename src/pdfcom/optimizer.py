from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from itertools import permutations
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterator

import pikepdf
import pymupdf


class Method(StrEnum):
    GHOSTSCRIPT = "ghostscript"
    PIKEPDF = "pikepdf"
    PYMUPDF = "pymupdf"


ALL_METHODS: tuple[Method, ...] = (
    Method.GHOSTSCRIPT,
    Method.PIKEPDF,
    Method.PYMUPDF,
)


@dataclass(frozen=True)
class Candidate:
    plan: tuple[Method, ...]
    path: Path
    size: int


@dataclass(frozen=True)
class OptimizationResult:
    source: Path
    original_size: int
    original_page_count: int
    best: Candidate
    attempted_plans: int
    failed_plans: dict[str, str]
    unavailable_methods: tuple[Method, ...]

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


def available_methods() -> tuple[tuple[Method, ...], tuple[Method, ...]]:
    unavailable = []
    methods = []
    for method in ALL_METHODS:
        if method is Method.GHOSTSCRIPT and shutil.which("gs") is None:
            unavailable.append(method)
        else:
            methods.append(method)
    return tuple(methods), tuple(unavailable)


def optimization_plans(methods: tuple[Method, ...]) -> list[tuple[Method, ...]]:
    plans: list[tuple[Method, ...]] = []
    for length in range(1, len(methods) + 1):
        plans.extend(permutations(methods, length))
    return plans


@contextmanager
def optimize_pdf(source: Path) -> Iterator[OptimizationResult]:
    with tempfile.TemporaryDirectory(prefix="pdfcom-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        yield _optimize_pdf(source, temp_dir)


def _optimize_pdf(source: Path, temp_dir: Path) -> OptimizationResult:
    source = source.resolve()
    original_size = source.stat().st_size
    original_page_count = page_count(source)
    best = Candidate((), source, original_size)
    methods, unavailable = available_methods()
    failures: dict[str, str] = {}
    attempted = 0

    for plan_index, plan in enumerate(optimization_plans(methods), start=1):
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
            if candidate_size < best.size:
                best = Candidate(plan, current, candidate_size)
        except Exception as exc:  # noqa: BLE001 - keep trying other compressors.
            failures[plan_label(plan)] = str(exc)

    return OptimizationResult(
        source=source,
        original_size=original_size,
        original_page_count=original_page_count,
        best=best,
        attempted_plans=attempted,
        failed_plans=failures,
        unavailable_methods=unavailable,
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
