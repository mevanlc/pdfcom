from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_STATS = Path.home() / "p/my/testpdfs-compressed/stats.json"
DEFAULT_OUT = Path("scratch/pdfcom-stats-summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a pdfcom stats JSON file.")
    parser.add_argument("stats", nargs="?", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = json.loads(args.stats.read_text())
    results = payload["results"]
    summary = summarize(payload, results)
    report = render_report(summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"\nwrote {args.out}")


def summarize(payload: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    input_count = int(payload["input_count"])
    recorded_count = len(results)
    original_total = sum(int(row["original_size"]) for row in results)
    best_total = sum(int(row["best_size"]) for row in results)
    reduced = [row for row in results if row["reduced"]]
    not_reduced = [row for row in results if not row["reduced"]]
    saved_bytes = [int(row["saved_bytes"]) for row in results]
    saved_percents = [float(row["saved_percent"]) for row in results]

    winner_counts = Counter(str(row["best_plan"]) for row in results)
    page_counts = [int(row["original_page_count"]) for row in results]
    sizes = [int(row["original_size"]) for row in results]
    failed_plan_counts: Counter[str] = Counter()
    failed_file_counts: Counter[str] = Counter()
    per_plan_ok = Counter()
    per_plan_present = Counter()
    per_plan_size_sum: Counter[str] = Counter()
    per_plan_best_gap: dict[str, list[int]] = defaultdict(list)

    qpdf_alone_delta = []
    best_over_qpdf_improvements = []
    best_gap_to_runner_up = []
    best_tie_counts = Counter()
    qpdf_alone_best_or_tied = 0
    qpdf_alone_within_1k = 0
    qpdf_alone_within_1pct = 0
    only_qpdf_beats_original = 0
    non_qpdf_stage_beats_qpdf = 0
    original_would_be_best = 0

    for row in results:
        original_size = int(row["original_size"])
        best_size = int(row["best_size"])
        plan_sizes = {
            str(plan["plan"]): int(plan["size"])
            for plan in row["plans"]
            if plan.get("ok") and plan.get("size") is not None
        }
        for plan in row["plans"]:
            plan_name = str(plan["plan"])
            per_plan_present[plan_name] += 1
            if plan.get("ok") and plan.get("size") is not None:
                size = int(plan["size"])
                per_plan_ok[plan_name] += 1
                per_plan_size_sum[plan_name] += size
                per_plan_best_gap[plan_name].append(size - best_size)
            else:
                failed_plan_counts[plan_name] += 1

        for plan_name, error in row.get("failed_plans", {}).items():
            failed_file_counts[f"{plan_name}: {str(error)[:120]}"] += 1

        candidates = {"original": original_size, **plan_sizes}
        sorted_sizes = sorted(candidates.items(), key=lambda item: item[1])
        best_value = sorted_sizes[0][1]
        winner_ties = [name for name, size in sorted_sizes if size == best_value]
        best_tie_counts[len(winner_ties)] += 1
        if len(sorted_sizes) > 1:
            runner_up_size = next((size for _, size in sorted_sizes if size > best_value), best_value)
            best_gap_to_runner_up.append(runner_up_size - best_value)
        if candidates["original"] == best_value:
            original_would_be_best += 1

        qpdf_size = plan_sizes.get("qpdf")
        if qpdf_size is not None:
            delta = qpdf_size - best_size
            qpdf_alone_delta.append(delta)
            if delta == 0:
                qpdf_alone_best_or_tied += 1
            if delta <= 1024:
                qpdf_alone_within_1k += 1
            if qpdf_size <= math.ceil(best_size * 1.01):
                qpdf_alone_within_1pct += 1
            if qpdf_size < original_size:
                best_over_qpdf_improvements.append(delta)
                if best_size < qpdf_size:
                    non_qpdf_stage_beats_qpdf += 1

        if qpdf_size is not None:
            any_other_reduced = any(
                size < original_size
                for name, size in plan_sizes.items()
                if name != "qpdf"
            )
            if qpdf_size < original_size and not any_other_reduced:
                only_qpdf_beats_original += 1

    plan_stats = []
    for plan, count in winner_counts.most_common():
        plan_stats.append(
            {
                "plan": plan,
                "wins": count,
                "share": count / recorded_count,
            }
        )

    per_plan = []
    for plan in sorted(per_plan_present):
        gaps = per_plan_best_gap.get(plan, [])
        per_plan.append(
            {
                "plan": plan,
                "ok": per_plan_ok[plan],
                "failed": failed_plan_counts[plan],
                "mean_gap": sum(gaps) / len(gaps) if gaps else None,
                "median_gap": median(gaps) if gaps else None,
                "within_1k": sum(1 for gap in gaps if gap <= 1024),
                "within_1pct": sum(
                    1
                    for row in results
                    for plan_result in row["plans"]
                    if plan_result.get("plan") == plan
                    and plan_result.get("ok")
                    and plan_result.get("size") is not None
                    and int(plan_result["size"]) <= math.ceil(int(row["best_size"]) * 1.01)
                ),
            }
        )

    return {
        "path_options": payload.get("options", {}),
        "input_count": input_count,
        "recorded_count": recorded_count,
        "missing_count": input_count - recorded_count,
        "original_total": original_total,
        "best_total": best_total,
        "total_saved": original_total - best_total,
        "total_saved_percent": (original_total - best_total) / original_total * 100 if original_total else 0,
        "reduced_count": len(reduced),
        "not_reduced_count": len(not_reduced),
        "median_saved_bytes": median(saved_bytes),
        "median_saved_percent": median(saved_percents),
        "page_count_median": median(page_counts),
        "page_count_max": max(page_counts),
        "size_median": median(sizes),
        "size_max": max(sizes),
        "winner_counts": plan_stats,
        "failed_plan_counts": failed_plan_counts.most_common(20),
        "failed_file_counts": failed_file_counts.most_common(20),
        "per_plan": sorted(per_plan, key=lambda item: (item["ok"], -item["failed"]), reverse=True),
        "qpdf_alone_best_or_tied": qpdf_alone_best_or_tied,
        "qpdf_alone_within_1k": qpdf_alone_within_1k,
        "qpdf_alone_within_1pct": qpdf_alone_within_1pct,
        "qpdf_alone_delta_sum": sum(qpdf_alone_delta),
        "qpdf_alone_delta_median": median(qpdf_alone_delta),
        "qpdf_alone_delta_p95": percentile(qpdf_alone_delta, 95),
        "non_qpdf_stage_beats_qpdf": non_qpdf_stage_beats_qpdf,
        "only_qpdf_beats_original": only_qpdf_beats_original,
        "original_would_be_best": original_would_be_best,
        "best_gap_to_runner_up_median": median(best_gap_to_runner_up),
        "best_gap_to_runner_up_p95": percentile(best_gap_to_runner_up, 95),
        "best_tie_counts": best_tie_counts,
    }


def percentile(values: list[int], pct: int) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[int(index)])
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def render_report(summary: dict[str, Any]) -> str:
    options = summary["path_options"]
    lines = [
        "# pdfcom stats summary",
        "",
        f"- plan spec: `{options.get('plan_spec')}`",
        f"- planned pipelines: {len(options.get('plans', []))}",
        f"- workers: {options.get('workers')}",
        f"- input count: {summary['input_count']:,}",
        f"- recorded results: {summary['recorded_count']:,}",
        f"- missing file-level results: {summary['missing_count']:,}",
        f"- total original size: {fmt_bytes(summary['original_total'])}",
        f"- total selected size: {fmt_bytes(summary['best_total'])}",
        f"- total saved: {fmt_bytes(summary['total_saved'])} ({summary['total_saved_percent']:.2f}%)",
        f"- reduced PDFs: {summary['reduced_count']:,} ({summary['reduced_count'] / summary['recorded_count']:.1%})",
        f"- copied originals: {summary['not_reduced_count']:,} ({summary['not_reduced_count'] / summary['recorded_count']:.1%})",
        f"- median per-file savings: {fmt_bytes(summary['median_saved_bytes'])} ({summary['median_saved_percent']:.2f}%)",
        f"- median input size: {fmt_bytes(summary['size_median'])}; max input size: {fmt_bytes(summary['size_max'])}",
        f"- median page count: {summary['page_count_median']}; max page count: {summary['page_count_max']}",
        "",
        "## Winning Plans",
        "",
        "| plan | wins | share |",
        "| --- | ---: | ---: |",
    ]
    for row in summary["winner_counts"]:
        lines.append(f"| `{row['plan']}` | {row['wins']:,} | {row['share']:.1%} |")

    lines.extend(
        [
            "",
            "## qpdf Alone Compared To Selected Best",
            "",
            f"- `qpdf` alone was selected best/tied best for {summary['qpdf_alone_best_or_tied']:,} PDFs "
            f"({summary['qpdf_alone_best_or_tied'] / summary['recorded_count']:.1%}).",
            f"- `qpdf` alone was within 1 KiB of selected best for {summary['qpdf_alone_within_1k']:,} PDFs "
            f"({summary['qpdf_alone_within_1k'] / summary['recorded_count']:.1%}).",
            f"- `qpdf` alone was within 1% of selected best for {summary['qpdf_alone_within_1pct']:,} PDFs "
            f"({summary['qpdf_alone_within_1pct'] / summary['recorded_count']:.1%}).",
            f"- Extra default-plan stages beyond `qpdf` recovered {fmt_bytes(summary['qpdf_alone_delta_sum'])} "
            "relative to using only `qpdf`.",
            f"- Median `qpdf`-to-best delta: {fmt_bytes(summary['qpdf_alone_delta_median'])}; "
            f"95th percentile delta: {fmt_bytes(summary['qpdf_alone_delta_p95'])}.",
            f"- A non-`qpdf` stage improved over `qpdf` on {summary['non_qpdf_stage_beats_qpdf']:,} PDFs "
            f"({summary['non_qpdf_stage_beats_qpdf'] / summary['recorded_count']:.1%}).",
            f"- `qpdf` alone was the only recorded plan to beat original on {summary['only_qpdf_beats_original']:,} PDFs.",
            "",
            "## Plan Health",
            "",
            "| plan | ok | failed | median gap to best | within 1 KiB | within 1% |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(summary["per_plan"], key=lambda item: item["plan"]):
        lines.append(
            f"| `{row['plan']}` | {row['ok']:,} | {row['failed']:,} | "
            f"{fmt_bytes(row['median_gap'] or 0)} | {row['within_1k']:,} | {row['within_1pct']:,} |"
        )

    lines.extend(
        [
            "",
            "## Failure Counts",
            "",
            "| failed plan | count |",
            "| --- | ---: |",
        ]
    )
    if summary["failed_plan_counts"]:
        for plan, count in summary["failed_plan_counts"]:
            lines.append(f"| `{plan}` | {count:,} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Close Results",
            "",
            f"- Original would be best or tied against recorded plans for {summary['original_would_be_best']:,} PDFs.",
            f"- Median winner-to-runner-up gap: {fmt_bytes(summary['best_gap_to_runner_up_median'])}.",
            f"- 95th percentile winner-to-runner-up gap: {fmt_bytes(summary['best_gap_to_runner_up_p95'])}.",
            f"- Exact best-size tie counts: {dict(summary['best_tie_counts'])}.",
            "",
        ]
    )
    return "\n".join(lines)


def fmt_bytes(value: float | int) -> str:
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{sign}{value:.0f} {unit}"
            return f"{sign}{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
