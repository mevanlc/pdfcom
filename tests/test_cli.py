from __future__ import annotations

import json
from pathlib import Path
import subprocess
import warnings

import pikepdf
import pytest

from pdfcom.cli import (
    CliOptions,
    InputPdf,
    ProcessReport,
    build_parser,
    discover_inputs,
    output_collision_errors,
    process_records,
    run,
)
from pdfcom.optimizer import (
    Candidate,
    DEFAULT_PLAN_SPEC,
    FULL_PERMUTATION_PLAN_SPEC,
    Method,
    OptimizationResult,
    PlanResult,
    _optimize_pdf,
    expand_plan_spec,
    optimization_plans,
    plan_label,
    run_pikepdf,
    run_qpdf,
    select_full_permutation_plans,
    select_plans,
)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7\n")


def test_discover_file_uses_basename_for_output(tmp_path: Path) -> None:
    pdf = tmp_path / "one.pdf"
    touch(pdf)

    records, errors = discover_inputs((pdf,), recurse=False)

    assert errors == []
    assert len(records) == 1
    assert records[0].source == pdf
    assert records[0].output_rel == Path("one.pdf")


def test_discover_dir_prefixes_outputs_with_input_dir(tmp_path: Path) -> None:
    directory = tmp_path / "docs"
    touch(directory / "one.pdf")
    touch(directory / "nested" / "two.pdf")

    records, errors = discover_inputs((directory,), recurse=False)

    assert errors == []
    assert [(record.source.name, record.output_rel) for record in records] == [
        ("one.pdf", Path("docs/one.pdf")),
    ]


def test_discover_recursive_dir_keeps_deep_prefix(tmp_path: Path) -> None:
    directory = tmp_path / "docs"
    touch(directory / "one.pdf")
    touch(directory / "nested" / "two.pdf")

    records, errors = discover_inputs((directory,), recurse=True)

    assert errors == []
    assert [record.output_rel for record in records] == [
        Path("docs/nested/two.pdf"),
        Path("docs/one.pdf"),
    ]


def test_duplicate_output_paths_are_reported(tmp_path: Path) -> None:
    left = tmp_path / "left" / "same.pdf"
    right = tmp_path / "right" / "same.pdf"
    touch(left)
    touch(right)

    records, errors = discover_inputs((left, right), recurse=False)

    assert errors == []
    assert output_collision_errors(records, tmp_path / "out")


def test_optimizer_plans_try_all_non_empty_permutations() -> None:
    plans = optimization_plans((Method.GHOSTSCRIPT, Method.PIKEPDF, Method.PYMUPDF))

    assert len(plans) == 15
    assert plan_label((Method.GHOSTSCRIPT, Method.PIKEPDF)) == "ghostscript+pikepdf"


def test_optimizer_plans_include_qpdf_in_full_permutations() -> None:
    plans = optimization_plans((Method.GHOSTSCRIPT, Method.PIKEPDF, Method.PYMUPDF, Method.QPDF))

    assert len(plans) == 64
    assert (Method.QPDF,) in plans
    assert (Method.GHOSTSCRIPT, Method.QPDF, Method.PYMUPDF, Method.PIKEPDF) in plans


def test_default_plan_spec_uses_qpdf_last_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pdfcom.optimizer.shutil.which", lambda command: f"/usr/bin/{command}")

    selection = select_plans()

    assert selection.spec == DEFAULT_PLAN_SPEC
    assert selection.unavailable_methods == ()
    assert selection.plans == (
        (Method.GHOSTSCRIPT, Method.PYMUPDF, Method.PIKEPDF, Method.QPDF),
        (Method.GHOSTSCRIPT, Method.PIKEPDF, Method.PYMUPDF, Method.QPDF),
        (Method.GHOSTSCRIPT, Method.PYMUPDF, Method.QPDF),
        (Method.GHOSTSCRIPT, Method.PIKEPDF, Method.QPDF),
        (Method.GHOSTSCRIPT, Method.QPDF),
        (Method.PYMUPDF, Method.PIKEPDF, Method.QPDF),
        (Method.PIKEPDF, Method.PYMUPDF, Method.QPDF),
        (Method.PYMUPDF, Method.QPDF),
        (Method.PIKEPDF, Method.QPDF),
        (Method.QPDF,),
    )


def test_full_permutation_selection_tries_all_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pdfcom.optimizer.shutil.which", lambda command: f"/usr/bin/{command}")

    selection = select_full_permutation_plans()

    assert selection.spec == FULL_PERMUTATION_PLAN_SPEC
    assert selection.unavailable_methods == ()
    assert selection.plans == tuple(
        optimization_plans((Method.GHOSTSCRIPT, Method.PIKEPDF, Method.PYMUPDF, Method.QPDF))
    )
    assert (Method.QPDF,) in selection.plans
    assert (Method.GHOSTSCRIPT, Method.QPDF, Method.PYMUPDF, Method.PIKEPDF) in selection.plans


def test_plan_selection_drops_unavailable_executables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pdfcom.optimizer.shutil.which", lambda command: None)

    selection = select_plans("gs?,{mu?,pike?},q")

    assert selection.unavailable_methods == (Method.GHOSTSCRIPT, Method.QPDF)
    assert selection.plans == (
        (Method.PYMUPDF, Method.PIKEPDF),
        (Method.PIKEPDF, Method.PYMUPDF),
        (Method.PYMUPDF,),
        (Method.PIKEPDF,),
    )


def test_plan_spec_minigrammar_examples() -> None:
    assert expand_plan_spec("{gs?,pike}") == (
        (Method.GHOSTSCRIPT, Method.PIKEPDF),
        (Method.PIKEPDF, Method.GHOSTSCRIPT),
        (Method.PIKEPDF,),
    )
    assert expand_plan_spec("(gs,pike?)") == (
        (Method.GHOSTSCRIPT, Method.PIKEPDF),
        (Method.GHOSTSCRIPT,),
    )
    assert expand_plan_spec("(gs,pike)?") == (
        (Method.GHOSTSCRIPT, Method.PIKEPDF),
    )
    assert expand_plan_spec("gs,{mu,pike},q") == (
        (Method.GHOSTSCRIPT, Method.PYMUPDF, Method.PIKEPDF, Method.QPDF),
        (Method.GHOSTSCRIPT, Method.PIKEPDF, Method.PYMUPDF, Method.QPDF),
    )


def test_parser_accepts_worker_count() -> None:
    namespace = build_parser().parse_args(["-w", "3", "-n", "one.pdf"])

    assert namespace.workers == 3


def test_parser_accepts_plan_spec() -> None:
    namespace = build_parser().parse_args(["-p", "q", "-n", "one.pdf"])

    assert namespace.plan_spec == "q"


def test_parser_accepts_full_permutation_shortcut() -> None:
    namespace = build_parser().parse_args(["-P", "-n", "one.pdf"])

    assert namespace.full_permutation is True


def test_parser_rejects_plan_spec_with_full_permutation_shortcut() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-P", "-p", "q", "-n", "one.pdf"])


def test_parser_accepts_stats_file() -> None:
    namespace = build_parser().parse_args(["-s", "stats.json", "-n", "one.pdf"])

    assert namespace.stats_file == Path("stats.json")


def test_parser_rejects_zero_workers() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-w", "0", "-n", "one.pdf"])


def test_process_records_streams_unordered_worker_results(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakePool:
        def __init__(self, processes: int) -> None:
            assert processes == 2

        def __enter__(self) -> "FakePool":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def imap_unordered(self, func: object, jobs: object) -> object:
            calls.append("imap_unordered")
            next(iter(jobs))
            return iter(
                (
                    ProcessReport(True, stdout=("second",)),
                    ProcessReport(True, stdout=("first",)),
                )
            )

    class FakeContext:
        Pool = FakePool

    monkeypatch.setattr("pdfcom.cli.multiprocessing.get_context", lambda name: FakeContext())
    records = [
        InputPdf(Path("one.pdf"), "one.pdf", Path("one.pdf")),
        InputPdf(Path("two.pdf"), "two.pdf", Path("two.pdf")),
    ]
    options = CliOptions(
        dry_run=True,
        recurse=False,
        inplace=False,
        output_dir=None,
        clobber=False,
        workers=2,
        plan_spec="q",
        plans=((Method.QPDF,),),
        unavailable_methods=(),
        stats_file=None,
        inputs=(),
    )

    reports = process_records(records, options)

    assert calls == []
    assert [report.stdout[0] for report in reports] == ["second", "first"]
    assert calls == ["imap_unordered"]


def test_pikepdf_page_copy_warning_disqualifies_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"original")

    monkeypatch.setattr("pdfcom.optimizer.page_count", lambda path: 1)

    def warn_on_pikepdf(method: Method, source: Path, output: Path) -> None:
        raise pikepdf.PageCopyWarning("orphaned form widgets")

    monkeypatch.setattr("pdfcom.optimizer.run_method", warn_on_pikepdf)

    result = _optimize_pdf(source, tmp_path, plans=((Method.PIKEPDF,),))

    assert result.best == Candidate((), source.resolve(), len(b"original"))
    assert result.failed_plans == {"pikepdf": "orphaned form widgets"}


def test_run_pikepdf_treats_page_copy_warning_as_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakePdf:
        def __enter__(self) -> "FakePdf":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def save(self, *args: object, **kwargs: object) -> None:
            warnings.warn("orphaned form widgets", pikepdf.PageCopyWarning)

    monkeypatch.setattr("pdfcom.optimizer.pikepdf.Pdf.open", lambda source: FakePdf())

    with pytest.raises(pikepdf.PageCopyWarning):
        run_pikepdf(tmp_path / "source.pdf", tmp_path / "output.pdf")


def test_run_qpdf_uses_lossless_compression_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stderr="")

    monkeypatch.setattr("pdfcom.optimizer.shutil.which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr("pdfcom.optimizer.subprocess.run", fake_run)
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"

    run_qpdf(source, output)

    assert commands == [
        [
            "/usr/bin/qpdf",
            "--compress-streams=y",
            "--decode-level=generalized",
            "--recompress-flate",
            "--compression-level=9",
            "--object-streams=generate",
            "--deterministic-id",
            str(source),
            str(output),
        ]
    ]


def test_run_writes_stats_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    touch(source)
    stats_file = tmp_path / "stats.json"

    class FakeOptimizePdf:
        def __init__(
            self,
            source: Path,
            *,
            plans: tuple[tuple[Method, ...], ...],
            unavailable_methods: tuple[Method, ...],
        ) -> None:
            assert plans == ((Method.QPDF,),)
            assert unavailable_methods == ()
            self.source = source

        def __enter__(self) -> OptimizationResult:
            original_size = self.source.stat().st_size
            best_size = original_size - 2
            return OptimizationResult(
                source=self.source,
                original_size=original_size,
                original_page_count=1,
                best=Candidate((Method.QPDF,), tmp_path / "best.pdf", best_size),
                attempted_plans=2,
                failed_plans={"pikepdf": "warning"},
                unavailable_methods=(Method.GHOSTSCRIPT,),
                plan_results=(
                    PlanResult((Method.PIKEPDF,), ok=False, error="warning"),
                    PlanResult((Method.QPDF,), ok=True, size=best_size, page_count=1),
                ),
            )

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr("pdfcom.cli.optimize_pdf", FakeOptimizePdf)
    monkeypatch.setattr("pdfcom.cli.select_plans", lambda spec: select_plans(spec))
    monkeypatch.setattr("pdfcom.optimizer.shutil.which", lambda command: f"/usr/bin/{command}")

    assert run(["-p", "q", "-n", "-s", str(stats_file), str(source)]) == 0

    payload = json.loads(stats_file.read_text())
    assert payload["version"] == 1
    assert payload["options"]["plan_spec"] == "q"
    assert payload["options"]["plans"] == [["qpdf"]]
    assert payload["input_count"] == 1
    assert payload["results"][0]["best_plan"] == "qpdf"
    plans = payload["results"][0]["plans"]
    assert plans[0] == [
        {
            "error": "warning",
            "methods": ["pikepdf"],
            "ok": False,
            "page_count": None,
            "plan": "pikepdf",
            "saved_bytes": None,
            "saved_percent": None,
            "size": None,
        },
    ][0]
    assert plans[1]["error"] is None
    assert plans[1]["methods"] == ["qpdf"]
    assert plans[1]["ok"] is True
    assert plans[1]["page_count"] == 1
    assert plans[1]["plan"] == "qpdf"
    assert plans[1]["saved_bytes"] == 2
    assert plans[1]["size"] == source.stat().st_size - 2
