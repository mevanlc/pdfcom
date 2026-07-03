from __future__ import annotations

from pathlib import Path

import pytest

from pdfcom.cli import build_parser, discover_inputs, output_collision_errors
from pdfcom.optimizer import Method, optimization_plans, plan_label


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


def test_parser_accepts_worker_count() -> None:
    namespace = build_parser().parse_args(["-w", "3", "-n", "one.pdf"])

    assert namespace.workers == 3


def test_parser_rejects_zero_workers() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-w", "0", "-n", "one.pdf"])
