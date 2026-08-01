from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = Path("scratch/pdfcom-features.jsonl")
DEFAULT_REPORT = Path("scratch/pdfcom-model-report.md")
TARGET_PLAN = "pymupdf+ghostscript+pikepdf+qpdf"
DEFAULT_PLAN_LABELS = {
    "ghostscript+pymupdf+pikepdf+qpdf",
    "ghostscript+pikepdf+pymupdf+qpdf",
    "ghostscript+pymupdf+qpdf",
    "ghostscript+pikepdf+qpdf",
    "ghostscript+qpdf",
    "pymupdf+pikepdf+qpdf",
    "pikepdf+pymupdf+qpdf",
    "pymupdf+qpdf",
    "pikepdf+qpdf",
    "qpdf",
    "original",
}
DEFAULT_PLAN_FEATURE_SUFFIXES = {label.replace("+", "__").replace("-", "_") for label in DEFAULT_PLAN_LABELS}


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.features)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"no rows found in {args.features}")

    frame = clean_frame(frame)
    target = frame["target_wins"].astype(bool).to_numpy()
    feature_columns = numeric_feature_columns(frame)
    x = frame[feature_columns]

    models = {
        "logistic_l2_balanced": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ),
        "extra_trees_balanced": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=13,
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        ),
    }

    cv = StratifiedKFold(n_splits=min(5, int(target.sum())), shuffle=True, random_state=13)
    model_reports = []
    scored = frame.copy()
    for name, model in models.items():
        probabilities = cross_val_predict(model, x, target, cv=cv, method="predict_proba")[:, 1]
        scored[f"score__{name}"] = probabilities
        model_reports.append(score_model(name, scored, probabilities, target))

    fit_model = models["extra_trees_balanced"]
    fit_model.fit(x, target)
    importances = fitted_importances(fit_model, feature_columns)
    thresholds = threshold_candidates(frame, feature_columns)
    ratio_summary = summarize_requested_ratios(frame)

    report = render_report(
        frame=frame,
        feature_columns=feature_columns,
        model_reports=model_reports,
        importances=importances,
        thresholds=thresholds,
        ratio_summary=ratio_summary,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(report)
    print(f"\nwrote {args.report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model pdfcom full-permutation target wins.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame[frame.get("error").isna()].copy() if "error" in frame else frame
    for column in frame.columns:
        if frame[column].dtype == bool:
            frame[column] = frame[column].astype(int)
    frame = frame.replace([math.inf, -math.inf], np.nan)
    return frame


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "target_wins",
        "target_plan",
        "target_size",
        "target_gain_bytes",
        "full_best_size",
        "full_best_plan",
        "default_best_size",
        "default_best_plan",
        "default_loss_bytes",
        "display",
        "source",
        "pdf_path",
        "error",
        "pikepdf_error",
        "pymupdf_error",
    }
    columns = []
    for column in frame.columns:
        if column in excluded:
            continue
        if leaks_full_permutation(column):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            if frame[column].notna().sum() >= 10 and frame[column].nunique(dropna=True) > 1:
                columns.append(column)
    return columns


def leaks_full_permutation(column: str) -> bool:
    if column.startswith("delta_full__"):
        return True
    if column.startswith("size__"):
        return column.removeprefix("size__") not in DEFAULT_PLAN_FEATURE_SUFFIXES
    if column.startswith("delta_default__"):
        return column.removeprefix("delta_default__") not in DEFAULT_PLAN_FEATURE_SUFFIXES
    return False


def score_model(name: str, frame: pd.DataFrame, probabilities: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-probabilities)
    total_gain = int(frame.loc[target, "target_gain_bytes"].fillna(0).sum())
    rows: dict[str, Any] = {
        "name": name,
        "average_precision": average_precision_score(target, probabilities),
        "roc_auc": roc_auc_score(target, probabilities),
        "topk": [],
    }
    for k in (19, 27, 50, 100, 171, 645):
        selected = frame.iloc[order[: min(k, len(order))]]
        hits = int(selected["target_wins"].sum())
        recovered = int(selected.loc[selected["target_wins"].astype(bool), "target_gain_bytes"].fillna(0).sum())
        rows["topk"].append(
            {
                "k": k,
                "hits": hits,
                "recovered_bytes": recovered,
                "recovered_share": recovered / total_gain if total_gain else 0,
            }
        )
    return rows


def fitted_importances(model: Any, feature_columns: list[str]) -> list[tuple[str, float]]:
    estimator = model.named_steps["extratreesclassifier"]
    values = estimator.feature_importances_
    return sorted(zip(feature_columns, values, strict=True), key=lambda item: item[1], reverse=True)


def threshold_candidates(frame: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
    target = frame["target_wins"].astype(bool)
    total_gain = int(frame.loc[target, "target_gain_bytes"].fillna(0).sum())
    candidates = []
    for column in feature_columns:
        series = frame[column]
        values = series.dropna()
        if values.nunique() < 4:
            continue
        quantiles = sorted(set(values.quantile(np.linspace(0.05, 0.95, 19)).tolist()))
        for threshold in quantiles:
            for direction in (">=", "<="):
                mask = series >= threshold if direction == ">=" else series <= threshold
                triggered = int(mask.sum())
                if triggered == 0 or triggered > max(400, len(frame) // 2):
                    continue
                hits = int((mask & target).sum())
                if hits == 0:
                    continue
                precision = hits / triggered
                recall = hits / int(target.sum())
                recovered = int(frame.loc[mask & target, "target_gain_bytes"].fillna(0).sum())
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
                candidates.append(
                    {
                        "feature": column,
                        "direction": direction,
                        "threshold": float(threshold),
                        "triggered": triggered,
                        "hits": hits,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "recovered_bytes": recovered,
                        "recovered_share": recovered / total_gain if total_gain else 0,
                    }
                )
    return sorted(
        candidates,
        key=lambda item: (item["recovered_share"], item["precision"], -item["triggered"]),
        reverse=True,
    )


def summarize_requested_ratios(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "text_to_image_bytes_ratio",
        "text_to_table_text_ratio",
        "table_text_to_text_ratio",
        "tex_latex_to_plain_text_ratio",
    ]
    target = frame["target_wins"].astype(bool)
    rows = []
    for column in columns:
        if column not in frame:
            continue
        rows.append(
            {
                "feature": column,
                "target_median": frame.loc[target, column].median(),
                "non_target_median": frame.loc[~target, column].median(),
                "target_p75": frame.loc[target, column].quantile(0.75),
                "non_target_p75": frame.loc[~target, column].quantile(0.75),
                "coverage": int(frame[column].notna().sum()),
            }
        )
    return rows


def render_report(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    model_reports: list[dict[str, Any]],
    importances: list[tuple[str, float]],
    thresholds: list[dict[str, Any]],
    ratio_summary: list[dict[str, Any]],
) -> str:
    target = frame["target_wins"].astype(bool)
    total_gain = int(frame.loc[target, "target_gain_bytes"].fillna(0).sum())
    lines = [
        "# pdfcom target-plan heuristic report",
        "",
        f"- rows: {len(frame)}",
        f"- features: {len(feature_columns)}",
        f"- target plan: `{TARGET_PLAN}`",
        f"- target wins: {int(target.sum())}",
        f"- recoverable default-plan delta in target wins: {total_gain:,} bytes",
        "",
        "## Cross-validated models",
    ]
    for item in model_reports:
        lines.extend(
            [
                "",
                f"### {item['name']}",
                f"- average precision: {item['average_precision']:.4f}",
                f"- ROC AUC: {item['roc_auc']:.4f}",
                "",
                "| top k | hits | recovered bytes | recovered share |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for row in item["topk"]:
            lines.append(
                f"| {row['k']} | {row['hits']} | {row['recovered_bytes']:,} | "
                f"{row['recovered_share']:.1%} |"
            )

    lines.extend(["", "## Requested Ratio Summary", ""])
    lines.append("| feature | coverage | target median | non-target median | target p75 | non-target p75 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in ratio_summary:
        lines.append(
            f"| `{row['feature']}` | {row['coverage']} | {fmt(row['target_median'])} | "
            f"{fmt(row['non_target_median'])} | {fmt(row['target_p75'])} | {fmt(row['non_target_p75'])} |"
        )

    lines.extend(["", "## Extra Trees Importances", ""])
    lines.append("| rank | feature | importance |")
    lines.append("| ---: | --- | ---: |")
    for rank, (feature, importance) in enumerate(importances[:25], start=1):
        lines.append(f"| {rank} | `{feature}` | {importance:.4f} |")

    lines.extend(["", "## Best Simple Thresholds", ""])
    lines.append("| rank | rule | triggered | hits | precision | recall | recovered share |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for rank, item in enumerate(thresholds[:25], start=1):
        lines.append(
            f"| {rank} | `{item['feature']} {item['direction']} {fmt(item['threshold'])}` | "
            f"{item['triggered']} | {item['hits']} | {item['precision']:.1%} | "
            f"{item['recall']:.1%} | {item['recovered_share']:.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
