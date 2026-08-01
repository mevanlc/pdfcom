from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any

import pikepdf
import pymupdf


DEFAULT_STATS = Path("/tmp/pdfcomstats.json")
DEFAULT_PDF_ROOTS = (Path("/tmp"), Path.cwd())
DEFAULT_OUT = Path("scratch/pdfcom-features.jsonl")

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
}
DEFAULT_BASELINE_LABELS = {"original", *DEFAULT_PLAN_LABELS}
TARGET_PLAN = "pymupdf+ghostscript+pikepdf+qpdf"

IMAGE_FILTER_BUCKETS = {
    "DCTDecode": "dct",
    "JPXDecode": "jpx",
    "JBIG2Decode": "jbig2",
    "CCITTFaxDecode": "ccitt",
    "FlateDecode": "flate",
}
COLORSPACE_BUCKETS = {
    "DeviceRGB": "rgb",
    "DeviceCMYK": "cmyk",
    "DeviceGray": "gray",
    "Indexed": "indexed",
    "ICCBased": "icc",
    "Separation": "separation",
    "DeviceN": "devicen",
}
TEX_METADATA_RE = re.compile(
    r"(tex|latex|pdftex|luatex|xetex|miktex|dvipdf|hyperref|kpathsea|beamer)",
    re.IGNORECASE,
)
TEX_FONT_RE = re.compile(
    r"(computer.?modern|latin.?modern|\bcm(r|mi|sy|ex|tt|ss|bx)|"
    r"\blm(roman|math|mono|sans)|mathjax|texgyre|stix|xits|cambria.?math)",
    re.IGNORECASE,
)
TEX_TEXT_RE = re.compile(
    r"(\\[A-Za-z]+|\$\$?|\\begin\{|\\end\{|LaTeX|TeX|pdfTeX|XeTeX|LuaTeX|\.tex\b)",
    re.IGNORECASE,
)


class JobTimeout(RuntimeError):
    pass


def main() -> None:
    args = parse_args()
    stats = json.loads(args.stats.read_text())
    records = stats["results"]
    if args.limit:
        records = records[: args.limit]

    existing = read_existing(args.out) if args.resume else set()
    jobs = []
    for index, record in enumerate(records):
        display = str(record["display"])
        if display in existing and not args.force:
            continue
        jobs.append(
            {
                "index": index,
                "record": record,
                "pdf_roots": [path.as_posix() for path in args.pdf_root],
                "table_mode": args.table_mode,
                "font_mode": args.font_mode,
                "max_table_pages": args.max_table_pages,
                "timeout_seconds": args.timeout_seconds,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and not args.force else "w"
    start = time.monotonic()
    done = 0
    failures = 0
    with args.out.open(mode) as out:
        if args.workers == 1:
            iterator = map(extract_one, jobs)
        else:
            context = multiprocessing.get_context("spawn")
            pool = context.Pool(processes=args.workers)
            iterator = pool.imap_unordered(extract_one, jobs, chunksize=args.chunksize)

        try:
            for result in iterator:
                done += 1
                if result.get("error"):
                    failures += 1
                out.write(json.dumps(result, sort_keys=True) + "\n")
                out.flush()
                if done == 1 or done % args.progress_every == 0 or result.get("error"):
                    elapsed = time.monotonic() - start
                    rate = done / elapsed if elapsed else 0
                    print(
                        f"{done:5d}/{len(jobs):5d} "
                        f"errors={failures:4d} "
                        f"rate={rate:5.2f}/s "
                        f"{result.get('display', '<unknown>')}",
                        flush=True,
                    )
        finally:
            if args.workers != 1:
                pool.close()
                pool.join()

    print(f"wrote {done} rows to {args.out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PDF features for pdfcom plan analysis.")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--pdf-root", type=Path, action="append", default=list(DEFAULT_PDF_ROOTS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("-w", "--workers", type=positive_int, default=max(1, min(os.cpu_count() or 1, 12)))
    parser.add_argument("--chunksize", type=positive_int, default=1)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--timeout-seconds", type=positive_int, default=90)
    parser.add_argument("--progress-every", type=positive_int, default=25)
    parser.add_argument(
        "--table-mode",
        choices=("off", "all", "interesting"),
        default="all",
        help="extract find_tables() text for all files, no files, or only target/default-loss files",
    )
    parser.add_argument(
        "--font-mode",
        choices=("off", "all", "interesting"),
        default="all",
        help="extract span-font text features for all files, no files, or only target/default-loss files",
    )
    parser.add_argument(
        "--max-table-pages",
        type=int,
        default=0,
        help="0 means all pages; positive values cap table extraction per PDF",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def read_existing(path: Path) -> set[str]:
    displays = set()
    if not path.exists():
        return displays
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                displays.add(json.loads(line)["display"])
            except Exception:
                continue
    return displays


def extract_one(job: dict[str, Any]) -> dict[str, Any]:
    timeout_seconds = int(job["timeout_seconds"])
    if timeout_seconds:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

    record = job["record"]
    row = base_row(record)
    try:
        path = resolve_pdf_path(record, [Path(value) for value in job["pdf_roots"]])
        row["pdf_path"] = path.as_posix()
        row.update(extract_pikepdf_features(path))
        include_tables = should_extract_deep_features(record, str(job["table_mode"]))
        include_fonts = should_extract_deep_features(record, str(job["font_mode"]))
        row.update(
            extract_pymupdf_features(
                path,
                include_tables=include_tables,
                include_fonts=include_fonts,
                max_table_pages=int(job["max_table_pages"]),
            )
        )
        add_ratios(row)
    except Exception as exc:  # noqa: BLE001 - keep corpus extraction moving.
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if timeout_seconds:
            signal.alarm(0)
    return row


def timeout_handler(signum: int, frame: object) -> None:
    raise JobTimeout("timed out while extracting PDF features")


def base_row(record: dict[str, Any]) -> dict[str, Any]:
    original_size = int(record["original_size"])
    plan_sizes = {
        str(plan["plan"]): plan["size"]
        for plan in record.get("plans", [])
        if plan.get("ok") and plan.get("size") is not None
    }
    default_sizes = {"original": original_size}
    default_sizes.update(
        {
            plan: size
            for plan, size in plan_sizes.items()
            if plan in DEFAULT_PLAN_LABELS
        }
    )
    default_best_plan = None
    default_best_size = None
    if default_sizes:
        default_best_plan, default_best_size = min(default_sizes.items(), key=lambda item: item[1])

    full_best_plan = str(record["best_plan"])
    full_best_size = int(record["best_size"])
    target_size = plan_sizes.get(TARGET_PLAN)
    default_loss_bytes = (
        max(int(default_best_size) - full_best_size, 0)
        if default_best_size is not None
        else None
    )
    target_gain_bytes = (
        max(int(default_best_size) - int(target_size), 0)
        if default_best_size is not None and target_size is not None
        else None
    )
    qpdf_size = plan_sizes.get("qpdf")

    row: dict[str, Any] = {
        "display": record["display"],
        "source": record["source"],
        "original_size": original_size,
        "original_page_count": int(record["original_page_count"]),
        "full_best_plan": full_best_plan,
        "full_best_size": full_best_size,
        "default_best_plan": default_best_plan,
        "default_best_size": default_best_size,
        "default_loss_bytes": default_loss_bytes,
        "target_plan": TARGET_PLAN,
        "target_wins": full_best_plan == TARGET_PLAN,
        "target_size": target_size,
        "target_gain_bytes": target_gain_bytes,
        "qpdf_size": qpdf_size,
        "qpdf_delta_to_default": (
            int(qpdf_size) - int(default_best_size)
            if qpdf_size is not None and default_best_size is not None
            else None
        ),
        "gs4_default_wins": default_best_plan
        in {
            "ghostscript+pymupdf+pikepdf+qpdf",
            "ghostscript+pikepdf+pymupdf+qpdf",
        },
        "default_best_is_original": default_best_plan == "original",
        "default_best_uses_ghostscript": bool(default_best_plan and "ghostscript" in default_best_plan),
        "default_best_uses_pymupdf": bool(default_best_plan and "pymupdf" in default_best_plan),
        "default_best_uses_pikepdf": bool(default_best_plan and "pikepdf" in default_best_plan),
        "default_best_uses_qpdf": bool(default_best_plan and "qpdf" in default_best_plan),
        "default_best_method_count": 0 if default_best_plan == "original" else len(str(default_best_plan).split("+")),
    }
    for plan in sorted(DEFAULT_BASELINE_LABELS):
        row[f"default_best__{safe_feature_name(plan)}"] = default_best_plan == plan

    for plan in (
        "original",
        "ghostscript",
        "pikepdf",
        "pymupdf",
        "qpdf",
        "ghostscript+qpdf",
        "pymupdf+qpdf",
        "pikepdf+qpdf",
        "ghostscript+pymupdf+qpdf",
        "pymupdf+pikepdf+qpdf",
        "pikepdf+pymupdf+qpdf",
        "ghostscript+pikepdf+qpdf",
        "ghostscript+pymupdf+pikepdf+qpdf",
        "ghostscript+pikepdf+pymupdf+qpdf",
        TARGET_PLAN,
    ):
        size = original_size if plan == "original" else plan_sizes.get(plan)
        row[f"size__{safe_feature_name(plan)}"] = size
        if size is not None:
            row[f"delta_default__{safe_feature_name(plan)}"] = int(size) - int(default_best_size)
            row[f"delta_full__{safe_feature_name(plan)}"] = int(size) - full_best_size
    return row


def safe_feature_name(value: str) -> str:
    return value.replace("+", "__").replace("-", "_")


def resolve_pdf_path(record: dict[str, Any], roots: list[Path]) -> Path:
    source = Path(str(record["source"]))
    candidates = []
    if source.is_absolute():
        candidates.append(source)
    candidates.append(Path.cwd() / source)
    for root in roots:
        candidates.append(root / source)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not resolve {source}; tried {', '.join(map(str, candidates))}")


def should_extract_deep_features(record: dict[str, Any], mode: str) -> bool:
    if mode == "off":
        return False
    if mode == "all":
        return True
    return str(record["best_plan"]) == TARGET_PLAN or default_loses(record)


def default_loses(record: dict[str, Any]) -> bool:
    plan_sizes = {
        str(plan["plan"]): plan["size"]
        for plan in record.get("plans", [])
        if plan.get("ok") and plan.get("size") is not None
    }
    included = [int(record["original_size"])]
    included.extend(size for plan, size in plan_sizes.items() if plan in DEFAULT_PLAN_LABELS)
    return bool(included and int(record["best_size"]) < min(included))


def extract_pikepdf_features(path: Path) -> dict[str, Any]:
    features: dict[str, Any] = {
        "pikepdf_error": None,
        "pdf_object_count": None,
        "pdf_stream_count": 0,
        "pdf_stream_length_total": 0,
        "image_stream_count": 0,
        "image_stream_length_total": 0,
        "image_pixel_total": 0,
        "image_max_pixels": 0,
        "widget_annotation_count": 0,
        "acroform_field_count": 0,
        "acroform_present": False,
    }
    filter_counts: Counter[str] = Counter()
    filter_lengths: Counter[str] = Counter()
    colorspaces: Counter[str] = Counter()

    try:
        with pikepdf.Pdf.open(path) as pdf:
            features["pdf_object_count"] = len(pdf.objects)
            features.update(pdf_metadata_features(pdf))
            root = pdf.Root
            acroform = root.get("/AcroForm")
            if acroform is not None:
                features["acroform_present"] = True
                fields = acroform.get("/Fields", [])
                try:
                    features["acroform_field_count"] = len(fields)
                except Exception:
                    features["acroform_field_count"] = 0

            for page in pdf.pages:
                annots = page.obj.get("/Annots", [])
                try:
                    for annot in annots:
                        if pdf_name(annot.get("/Subtype")) == "Widget":
                            features["widget_annotation_count"] += 1
                except Exception:
                    pass

            for obj in pdf.objects:
                if not isinstance(obj, pikepdf.Stream):
                    continue
                length = stream_length(obj)
                features["pdf_stream_count"] += 1
                features["pdf_stream_length_total"] += length
                if pdf_name(obj.get("/Subtype")) != "Image":
                    continue

                features["image_stream_count"] += 1
                features["image_stream_length_total"] += length
                filter_bucket = image_filter_bucket(obj.get("/Filter"))
                filter_counts[filter_bucket] += 1
                filter_lengths[filter_bucket] += length
                colorspaces[colorspace_bucket(obj.get("/ColorSpace"))] += 1

                width = int_or_zero(obj.get("/Width"))
                height = int_or_zero(obj.get("/Height"))
                pixels = width * height
                features["image_pixel_total"] += pixels
                features["image_max_pixels"] = max(int(features["image_max_pixels"]), pixels)
    except Exception as exc:  # noqa: BLE001
        features["pikepdf_error"] = f"{type(exc).__name__}: {exc}"

    for bucket in ("dct", "jpx", "jbig2", "ccitt", "flate", "unfiltered", "other"):
        features[f"image_filter_count__{bucket}"] = filter_counts[bucket]
        features[f"image_filter_length__{bucket}"] = filter_lengths[bucket]
    for bucket in ("rgb", "cmyk", "gray", "indexed", "icc", "separation", "devicen", "other"):
        features[f"image_colorspace_count__{bucket}"] = colorspaces[bucket]
    return features


def pdf_metadata_features(pdf: pikepdf.Pdf) -> dict[str, Any]:
    text_parts = []
    for key, value in dict(pdf.docinfo).items():
        text_parts.append(str(key))
        text_parts.append(str(value))
    metadata_text = "\n".join(text_parts)
    return {
        "pdf_docinfo_bytes": len(metadata_text.encode("utf-8", "ignore")),
        "pdf_docinfo_tex_marker_bytes": marker_bytes(metadata_text, TEX_METADATA_RE),
        "pdf_docinfo_has_tex_marker": bool(TEX_METADATA_RE.search(metadata_text)),
    }


def stream_length(stream: pikepdf.Stream) -> int:
    try:
        return int(stream.get("/Length", 0))
    except Exception:
        return 0


def image_filter_bucket(value: object) -> str:
    names = pdf_names(value)
    if not names:
        return "unfiltered"
    buckets = [IMAGE_FILTER_BUCKETS.get(name, "other") for name in names]
    for bucket in ("jbig2", "ccitt", "jpx", "dct", "flate"):
        if bucket in buckets:
            return bucket
    return "other"


def colorspace_bucket(value: object) -> str:
    names = pdf_names(value)
    if not names:
        return "other"
    return COLORSPACE_BUCKETS.get(names[0], "other")


def pdf_names(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, pikepdf.Array):
        return [pdf_name(item) for item in value if pdf_name(item)]
    name = pdf_name(value)
    return [name] if name else []


def pdf_name(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return text[1:] if text.startswith("/") else text


def int_or_zero(value: object) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def extract_pymupdf_features(
    path: Path,
    *,
    include_tables: bool,
    include_fonts: bool,
    max_table_pages: int,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "pymupdf_error": None,
        "plain_text_bytes": 0,
        "plain_text_chars": 0,
        "plain_text_words": 0,
        "table_text_bytes": 0,
        "table_count": 0,
        "tex_latex_marker_bytes": 0,
        "tex_latex_font_text_bytes": 0,
        "metadata_tex_marker_bytes": 0,
        "metadata_has_tex_marker": False,
        "font_span_count": 0,
        "font_unique_count": 0,
    }
    fonts: set[str] = set()

    try:
        with pymupdf.open(path) as document:
            metadata_text = "\n".join(str(value or "") for value in document.metadata.values())
            features["metadata_tex_marker_bytes"] = marker_bytes(metadata_text, TEX_METADATA_RE)
            features["metadata_has_tex_marker"] = bool(TEX_METADATA_RE.search(metadata_text))

            for page_index, page in enumerate(document):
                if include_fonts:
                    text_bytes, chars, tex_font_bytes, span_count = extract_span_text(page, fonts)
                    features["plain_text_bytes"] += text_bytes
                    features["plain_text_chars"] += chars
                    features["tex_latex_font_text_bytes"] += tex_font_bytes
                    features["font_span_count"] += span_count
                else:
                    text = page.get_text("text")
                    features["plain_text_bytes"] += len(text.encode("utf-8", "ignore"))
                    features["plain_text_chars"] += len(text)
                    features["tex_latex_marker_bytes"] += marker_bytes(text, TEX_TEXT_RE)

                try:
                    features["plain_text_words"] += len(page.get_text("words"))
                except Exception:
                    pass

                if include_tables and (max_table_pages <= 0 or page_index < max_table_pages):
                    table_bytes, table_count = extract_table_text(page)
                    features["table_text_bytes"] += table_bytes
                    features["table_count"] += table_count

            features["font_unique_count"] = len(fonts)
    except Exception as exc:  # noqa: BLE001
        features["pymupdf_error"] = f"{type(exc).__name__}: {exc}"

    return features


def extract_span_text(page: pymupdf.Page, fonts: set[str]) -> tuple[int, int, int, int]:
    text_bytes = 0
    chars = 0
    tex_font_bytes = 0
    span_count = 0
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = str(span.get("text", ""))
                font = str(span.get("font", ""))
                if font:
                    fonts.add(font)
                byte_count = len(text.encode("utf-8", "ignore"))
                text_bytes += byte_count
                chars += len(text)
                span_count += 1
                if font and TEX_FONT_RE.search(font):
                    tex_font_bytes += byte_count
                tex_font_bytes += marker_bytes(text, TEX_TEXT_RE)
    return text_bytes, chars, tex_font_bytes, span_count


def extract_table_text(page: pymupdf.Page) -> tuple[int, int]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tables = page.find_tables()
    total = 0
    count = 0
    for table in tables.tables:
        count += 1
        try:
            rows = table.extract()
        except Exception:
            continue
        parts = []
        for row in rows:
            for cell in row:
                if cell:
                    parts.append(str(cell))
        total += len("\n".join(parts).encode("utf-8", "ignore"))
    return total, count


def marker_bytes(text: str, pattern: re.Pattern[str]) -> int:
    return sum(len(match.group(0).encode("utf-8", "ignore")) for match in pattern.finditer(text))


def add_ratios(row: dict[str, Any]) -> None:
    ratio(row, "text_to_image_bytes_ratio", row.get("plain_text_bytes"), row.get("image_stream_length_total"))
    ratio(row, "text_to_table_text_ratio", row.get("plain_text_bytes"), row.get("table_text_bytes"))
    ratio(row, "table_text_to_text_ratio", row.get("table_text_bytes"), row.get("plain_text_bytes"))
    tex_bytes = (
        int(row.get("tex_latex_font_text_bytes") or 0)
        + int(row.get("tex_latex_marker_bytes") or 0)
        + int(row.get("metadata_tex_marker_bytes") or 0)
        + int(row.get("pdf_docinfo_tex_marker_bytes") or 0)
    )
    row["tex_latex_bytes"] = tex_bytes
    ratio(row, "tex_latex_to_plain_text_ratio", tex_bytes, row.get("plain_text_bytes"))
    ratio(row, "image_bytes_per_page", row.get("image_stream_length_total"), row.get("original_page_count"))
    ratio(row, "text_bytes_per_page", row.get("plain_text_bytes"), row.get("original_page_count"))
    ratio(row, "stream_bytes_to_file_size_ratio", row.get("pdf_stream_length_total"), row.get("original_size"))
    ratio(row, "image_bytes_to_file_size_ratio", row.get("image_stream_length_total"), row.get("original_size"))


def ratio(row: dict[str, Any], name: str, numerator: object, denominator: object) -> None:
    try:
        num = float(numerator or 0)
        den = float(denominator or 0)
    except Exception:
        row[name] = None
        return
    if den == 0:
        row[name] = None if num == 0 else math.inf
    else:
        row[name] = num / den


if __name__ == "__main__":
    main()
