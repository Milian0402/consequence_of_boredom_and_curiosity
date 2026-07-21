#!/usr/bin/env python3
import csv
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "hf" / "sgemm-benchmarks-dataset"
SPACE_DIR = ROOT / "hf" / "sgemm-dashboard-space"
DATA_DIR = DATASET_DIR / "data"
TIMELINE_PATH = ROOT / "docs" / "optimization-timeline.md"
ARTIFACT_DIR = Path(
    os.environ.get("COB_HF_ARTIFACT_DIR", ROOT / "hf" / "source-artifacts")
).expanduser()


def artifact_path(relative_path):
    return ARTIFACT_DIR / relative_path


def portable_source_path(path):
    try:
        return str(path.relative_to(ARTIFACT_DIR))
    except ValueError:
        return path.name


BENCHMARK_SOURCES = [
    (
        "hf-fresh-trophy-r21-i8",
        artifact_path("cob-hf-interesting-20260527/trophy_sweep.csv"),
        "Fresh May 27 headline sweep: repeat-21, iters=8, route-aware, single-thread env pins.",
    ),
    (
        "current-post-sme-r21-i8",
        artifact_path("cob-current-post-sme-r21-i8.csv"),
        "Current May 27 post-SME-reuse cooled target sweep.",
    ),
    (
        "current-targets-r21-i8",
        artifact_path("cob-current-targets-r21-i8.csv"),
        "May 27 cooled target sweep before high-K SME reuse extension.",
    ),
    (
        "qos-cooldown-r31",
        artifact_path("cob-targets-current-qos-cooldown-r31.csv"),
        "May 27 QoS/cooldown target sweep.",
    ),
    (
        "mpgemm-calibration-full-r101-i4-cob",
        artifact_path("cob-mpgemm-calibration-20260527-full-r101-i4/cob-one-shot.csv"),
        "May 27 focused MpGEMM FP32 row_sgemm calibration, COB rows.",
    ),
]


FRESH_PAIRED_SOURCES = [
    (
        "hf-fresh-current-vs-head-direct-r61-i16",
        artifact_path("cob-hf-interesting-20260527/current_vs_head_direct.txt"),
        "current_vs_repo_head",
        {"A": "repo_head", "B": "current"},
        "Fresh current source against clean HEAD. Direct one-shot mode.",
    ),
    (
        "hf-fresh-current-vs-head-direct-r61-i16",
        artifact_path("cob-hf-interesting-20260527/current_vs_head_direct_768_1024.txt"),
        "current_vs_repo_head",
        {"A": "repo_head", "B": "current"},
        "Fresh current source against clean HEAD for m=768/1024 high-K SME reuse rows.",
    ),
    (
        "hf-fresh-accelerate-vs-cob-direct-r61-i16",
        artifact_path("cob-hf-interesting-20260527/current_vs_accelerate_direct.txt"),
        "accelerate_vs_cob",
        {"COB": "COB", "Accelerate": "Accelerate"},
        "Fresh Apple Accelerate direct one-shot paired comparison.",
    ),
]


ROUTE_BY_SHAPE = {
    "512x1216x3072": "sme_medium_reuse",
    "768x512x4096": "sme_large_reuse",
    "768x768x4096": "sme_large_reuse",
    "1024x512x4096": "sme_large_reuse",
    "1024x768x4096": "sme_large_reuse",
    "1536x512x4096": "sme_large_reuse",
    "1536x768x4096": "sme_large_reuse",
    "2048x512x4096": "sme_large_reuse",
    "2048x768x4096": "sme_large_reuse",
}


PAIRED_CONFIRMATIONS = [
    {
        "run_id": "source-vs-head-512x1216x3072-prefetch",
        "date": "2026-05-27",
        "comparison": "candidate_vs_clean_head",
        "shape": "512x1216x3072",
        "m": 512,
        "n": 1216,
        "k": 3072,
        "mode": "one-shot",
        "route": "sme_medium_reuse",
        "ratio_label": "candidate/baseline",
        "median_ratio": 1.1829,
        "bootstrap_low": 1.1489,
        "bootstrap_high": 1.2105,
        "holdout_median": 1.1845,
        "repeats": 101,
        "iters": 16,
        "cooldown_us": 20000,
        "notes": "Accepted exact SME source-B reuse route with source-B prefetch distance 16.",
    },
    {
        "run_id": "accelerate-post-sme-large-768-1024",
        "date": "2026-05-27",
        "comparison": "accelerate_vs_cob",
        "shape": "768x768x4096",
        "m": 768,
        "n": 768,
        "k": 4096,
        "mode": "one-shot",
        "route": "sme_large_reuse",
        "ratio_label": "Accelerate/COB",
        "median_ratio": 0.9968,
        "bootstrap_low": 0.9732,
        "bootstrap_high": 1.0024,
        "holdout_median": 0.9938,
        "repeats": 61,
        "iters": 16,
        "cooldown_us": 20000,
        "notes": "Neutral after SME source-B reuse extension.",
    },
    {
        "run_id": "accelerate-post-sme-large-768-1024",
        "date": "2026-05-27",
        "comparison": "accelerate_vs_cob",
        "shape": "1024x512x4096",
        "m": 1024,
        "n": 512,
        "k": 4096,
        "mode": "one-shot",
        "route": "sme_large_reuse",
        "ratio_label": "Accelerate/COB",
        "median_ratio": 1.0517,
        "bootstrap_low": 1.0340,
        "bootstrap_high": 1.0740,
        "holdout_median": 1.0461,
        "repeats": 61,
        "iters": 16,
        "cooldown_us": 20000,
        "notes": "Accelerate still ahead in this paired run.",
    },
    {
        "run_id": "accelerate-post-sme-large-768-1024",
        "date": "2026-05-27",
        "comparison": "accelerate_vs_cob",
        "shape": "768x512x4096",
        "m": 768,
        "n": 512,
        "k": 4096,
        "mode": "one-shot",
        "route": "sme_large_reuse",
        "ratio_label": "Accelerate/COB",
        "median_ratio": 1.0240,
        "bootstrap_low": 1.0051,
        "bootstrap_high": 1.0257,
        "holdout_median": 1.0240,
        "repeats": 61,
        "iters": 16,
        "cooldown_us": 20000,
        "notes": "Small remaining Accelerate lead in this paired run.",
    },
    {
        "run_id": "accelerate-post-sme-large-768-1024",
        "date": "2026-05-27",
        "comparison": "accelerate_vs_cob",
        "shape": "1024x768x4096",
        "m": 1024,
        "n": 768,
        "k": 4096,
        "mode": "one-shot",
        "route": "sme_large_reuse",
        "ratio_label": "Accelerate/COB",
        "median_ratio": 0.9974,
        "bootstrap_low": 0.9313,
        "bootstrap_high": 1.0158,
        "holdout_median": 0.9936,
        "repeats": 61,
        "iters": 16,
        "cooldown_us": 20000,
        "notes": "Neutral after SME source-B reuse extension.",
    },
    {
        "run_id": "accelerate-post-m2048-extension",
        "date": "2026-05-27",
        "comparison": "accelerate_vs_cob",
        "shape": "2048x512x4096",
        "m": 2048,
        "n": 512,
        "k": 4096,
        "mode": "one-shot",
        "route": "sme_large_reuse",
        "ratio_label": "Accelerate/COB",
        "median_ratio": 1.0703,
        "bootstrap_low": 1.0124,
        "bootstrap_high": 1.1517,
        "holdout_median": 1.0703,
        "repeats": 61,
        "iters": 16,
        "cooldown_us": 20000,
        "notes": "Clearest current proprietary baseline target; high sample CV.",
    },
    {
        "run_id": "accelerate-post-m2048-extension",
        "date": "2026-05-27",
        "comparison": "accelerate_vs_cob",
        "shape": "2048x768x4096",
        "m": 2048,
        "n": 768,
        "k": 4096,
        "mode": "one-shot",
        "route": "sme_large_reuse",
        "ratio_label": "Accelerate/COB",
        "median_ratio": 1.0156,
        "bootstrap_low": 0.9757,
        "bootstrap_high": 1.0775,
        "holdout_median": 0.9937,
        "repeats": 61,
        "iters": 16,
        "cooldown_us": 20000,
        "notes": "Neutral after m=2048 route extension.",
    },
    {
        "run_id": "accelerate-post-m2048-extension",
        "date": "2026-05-27",
        "comparison": "accelerate_vs_cob",
        "shape": "1536x768x4096",
        "m": 1536,
        "n": 768,
        "k": 4096,
        "mode": "one-shot",
        "route": "sme_large_reuse",
        "ratio_label": "Accelerate/COB",
        "median_ratio": 0.9636,
        "bootstrap_low": 0.9435,
        "bootstrap_high": 0.9939,
        "holdout_median": 0.9549,
        "repeats": 61,
        "iters": 16,
        "cooldown_us": 20000,
        "notes": "COB-favored in latest paired Accelerate run.",
    },
]


RESEARCH_ATTEMPTS = [
    {
        "date": "2026-05-27",
        "result": "accepted",
        "name": "exact 512x1216x3072 SME source-B reuse",
        "shape_scope": "512x1216x3072",
        "headline": "Original repeat-101 current-vs-clean-head median 1.1829x; fresh repeat-61 rerun measured 1.1135x.",
        "why_interesting": "Largest clean one-shot source improvement from this pass, and it keeps Accelerate roughly neutral.",
        "notes": "Uses NC=256, KC=1024, and source-B prefetch distance 16.",
    },
    {
        "date": "2026-05-27",
        "result": "accepted",
        "name": "m=768/1024 k=4096 SME source-B reuse",
        "shape_scope": "768/1024 x 512/768 x 4096",
        "headline": "Fresh repeat-61 current-vs-head medians were 1.0298x to 1.0456x, with noisy samples.",
        "why_interesting": "Shows the route generalizes beyond one exact shape, but does not close Accelerate yet.",
        "notes": "Same NC=256, KC=1024, source-B prefetch distance 16.",
    },
    {
        "date": "2026-05-27",
        "result": "mixed",
        "name": "m=1536/2048 k=4096 SME source-B reuse extension",
        "shape_scope": "1536/2048 x 512/768 x 4096",
        "headline": "Fresh repeat-61 current-vs-head rerun regressed these rows, while earlier Accelerate triage was mixed.",
        "why_interesting": "Useful because it flags a likely overextended route rather than hiding a weak result.",
        "notes": "Keep under investigation before using it as a positive claim.",
    },
    {
        "date": "2026-05-27",
        "result": "rejected",
        "name": "512x1216x3072 block-size retunes",
        "shape_scope": "512x1216x3072",
        "headline": "NC=128/192/512 and KC=512/2048/3072 failed to beat the accepted NC=256, KC=1024 route.",
        "why_interesting": "Shows the final route was not picked from a single lucky knob.",
        "notes": "KC=1536 looked tempting against the old baseline but lost candidate-vs-candidate.",
    },
    {
        "date": "2026-05-27",
        "result": "rejected",
        "name": "source-B prefetch distances away from 16",
        "shape_scope": "512x1216x3072 and high-K SME reuse rows",
        "headline": "Distances 8/32/64 did not beat the distance-16 helper.",
        "why_interesting": "Documents a real locality search rather than an unexplained magic constant.",
        "notes": "Fresh route keeps distance 16.",
    },
    {
        "date": "2026-05-27",
        "result": "rejected",
        "name": "packed-B SME inner-loop unroll2",
        "shape_scope": "SME reuse route",
        "headline": "Passed correctness but did not clear A/B validation and weakened the earlier 512x1216x3072 route.",
        "why_interesting": "A classic obvious micro-optimization that the compiler/runtime already handled better.",
        "notes": "Left out of source.",
    },
    {
        "date": "2026-05-27",
        "result": "rejected",
        "name": "row-streaming AMX B packing",
        "shape_scope": "high-K medium rows",
        "headline": "Hard regression when SME was disabled, dropping tested rows to roughly 0.70x..0.79x.",
        "why_interesting": "Negative result narrows the remaining path: the issue is not fixed by a simple AMX row-stream pack rewrite.",
        "notes": "Left out of source.",
    },
    {
        "date": "2026-05-27",
        "result": "accepted",
        "name": "m=64 wide tuple-prefetch calibration",
        "shape_scope": "64 x long-N x K",
        "headline": "MpGEMM calibration includes COB wins on rows like 64x32768x512 and 64x24576x1536.",
        "why_interesting": "Shows the project is not only optimizing medium square-ish rows.",
        "notes": "See mpgemm_comparison.csv for source-available calibration rows.",
    },
]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def export_benchmarks():
    fieldnames = [
        "run_id",
        "source_path",
        "notes",
        "kind",
        "implementation",
        "m",
        "n",
        "k",
        "shape",
        "best_throughput",
        "median_throughput",
        "unit",
        "best_seconds",
        "median_seconds",
        "checksum",
        "route",
    ]
    rows = []
    for run_id, source, notes in BENCHMARK_SOURCES:
        if not source.exists():
            continue
        with source.open(newline="") as handle:
            for row in csv.DictReader(handle):
                m = row.get("m", "")
                n = row.get("n", "")
                k = row.get("k", "")
                out = dict(row)
                out.update(
                    {
                        "run_id": run_id,
                        "source_path": portable_source_path(source),
                        "notes": notes,
                        "shape": f"{m}x{n}x{k}" if m and n and k else "",
                    }
                )
                rows.append(out)
    target = DATA_DIR / "benchmark_results.csv"
    if rows or not target.exists():
        write_csv(target, rows, fieldnames)


def export_paired_confirmations():
    fieldnames = [
        "run_id",
        "date",
        "comparison",
        "shape",
        "m",
        "n",
        "k",
        "mode",
        "route",
        "ratio_label",
        "median_ratio",
        "bootstrap_low",
        "bootstrap_high",
        "holdout_median",
        "repeats",
        "iters",
        "cooldown_us",
        "notes",
    ]
    write_csv(DATA_DIR / "paired_confirmations.csv", PAIRED_CONFIRMATIONS, fieldnames)


def parse_shape_text(shape):
    m, n, k = shape.split("x")
    return int(m), int(n), int(k)


def relabel(name, label_map):
    return label_map.get(name, name)


def parse_paired_text(path, run_id, comparison, label_map, notes):
    if not path.exists():
        return []

    shape_line = re.compile(r"^(\d+x\d+x\d+) mode=([^ ]+) repeats=(\d+) iterations=(\d+)$")
    cooldown_line = re.compile(r"^\s+cooldown: (\d+) us after each paired sample$")
    impl_line = re.compile(
        r"^\s+(.+?) median\s+([0-9.]+) GF/s best\s+([0-9.]+) GF/s cv\s+([0-9.]+)%")
    paired_line = re.compile(
        r"^\s+paired (.+?)/(.+?) median ([0-9.]+)x mean-log ([0-9.]+)x cv\s+([0-9.]+)%"
        r"(?: bootstrap95 \[([0-9.]+)x, ([0-9.]+)x\])?"
        r" (.+?)-faster (\d+)/(\d+) sign-p ([0-9.eE+-]+)$")
    holdout_line = re.compile(
        r"^\s+split-half holdout (.+?)/(.+?) median ([0-9.]+)x mean-log ([0-9.]+)x cv\s+([0-9.]+)%"
        r"(?: bootstrap95 \[([0-9.]+)x, ([0-9.]+)x\])?"
        r" (.+?)-faster (\d+)/(\d+) sign-p ([0-9.eE+-]+)$")

    rows = []
    current = None
    for line in path.read_text().splitlines():
        shape_match = shape_line.match(line)
        if shape_match:
            if current is not None:
                rows.append(current)
            shape, mode, repeats, iterations = shape_match.groups()
            m, n, k = parse_shape_text(shape)
            current = {
                "run_id": run_id,
                "source_file": str(path),
                "date": "2026-05-27",
                "comparison": comparison,
                "shape": shape,
                "m": m,
                "n": n,
                "k": k,
                "mode": mode,
                "route": ROUTE_BY_SHAPE.get(shape, ""),
                "repeats": int(repeats),
                "iters": int(iterations),
                "cooldown_us": "",
                "notes": notes,
                "quality_note": "",
            }
            continue

        if current is None:
            continue

        cooldown_match = cooldown_line.match(line)
        if cooldown_match:
            current["cooldown_us"] = int(cooldown_match.group(1))
            continue

        impl_match = impl_line.match(line)
        if impl_match:
            label, median, best, cv = impl_match.groups()
            label = relabel(label, label_map)
            if "a_label" not in current:
                prefix = "a"
            else:
                prefix = "b"
            current[f"{prefix}_label"] = label
            current[f"{prefix}_median_gflops"] = median
            current[f"{prefix}_best_gflops"] = best
            current[f"{prefix}_cv_percent"] = cv
            continue

        paired_match = paired_line.match(line)
        if paired_match:
            numerator, denominator, median, mean_log, cv, low, high, faster_label, faster, total, sign_p = (
                paired_match.groups()
            )
            numerator = relabel(numerator, label_map)
            denominator = relabel(denominator, label_map)
            current["ratio_label"] = f"{numerator}/{denominator}"
            current["median_ratio"] = median
            current["mean_log_ratio"] = mean_log
            current["ratio_cv_percent"] = cv
            current["bootstrap_low"] = low or ""
            current["bootstrap_high"] = high or ""
            current["faster_label"] = relabel(faster_label, label_map)
            current["faster_count"] = faster
            current["paired_total"] = total
            current["sign_pvalue"] = sign_p
            continue

        holdout_match = holdout_line.match(line)
        if holdout_match:
            _, _, median, mean_log, cv, low, high, faster_label, faster, total, sign_p = holdout_match.groups()
            current["holdout_median"] = median
            current["holdout_mean_log_ratio"] = mean_log
            current["holdout_cv_percent"] = cv
            current["holdout_bootstrap_low"] = low or ""
            current["holdout_bootstrap_high"] = high or ""
            current["holdout_faster_label"] = relabel(faster_label, label_map)
            current["holdout_faster_count"] = faster
            current["holdout_total"] = total
            current["holdout_sign_pvalue"] = sign_p
            continue

        stripped = line.strip()
        if stripped.startswith("warning:") or stripped.startswith("note:"):
            current["quality_note"] = stripped

    if current is not None:
        rows.append(current)
    return rows


def export_fresh_paired_runs():
    fieldnames = [
        "run_id",
        "source_file",
        "date",
        "comparison",
        "shape",
        "m",
        "n",
        "k",
        "mode",
        "route",
        "a_label",
        "a_median_gflops",
        "a_best_gflops",
        "a_cv_percent",
        "b_label",
        "b_median_gflops",
        "b_best_gflops",
        "b_cv_percent",
        "ratio_label",
        "median_ratio",
        "mean_log_ratio",
        "ratio_cv_percent",
        "bootstrap_low",
        "bootstrap_high",
        "faster_label",
        "faster_count",
        "paired_total",
        "sign_pvalue",
        "holdout_median",
        "holdout_mean_log_ratio",
        "holdout_cv_percent",
        "holdout_bootstrap_low",
        "holdout_bootstrap_high",
        "holdout_faster_label",
        "holdout_faster_count",
        "holdout_total",
        "holdout_sign_pvalue",
        "repeats",
        "iters",
        "cooldown_us",
        "quality_note",
        "notes",
    ]
    rows = []
    for run_id, path, comparison, label_map, notes in FRESH_PAIRED_SOURCES:
        rows.extend(parse_paired_text(path, run_id, comparison, label_map, notes))
    target = DATA_DIR / "fresh_paired_runs.csv"
    if rows or not target.exists():
        write_csv(target, rows, fieldnames)


def export_validation_runs():
    rows = [
        {
            "run_id": "hf-fresh-make-test-20260527",
            "date": "2026-05-27",
            "command": "make test",
            "status": "passed",
            "reported_shapes": 636,
            "notes": "Local output: all GEMM tests passed (636 shapes).",
        }
    ]
    write_csv(
        DATA_DIR / "validation_runs.csv",
        rows,
        ["run_id", "date", "command", "status", "reported_shapes", "notes"],
    )


def export_research_attempts():
    write_csv(
        DATA_DIR / "research_attempts.csv",
        RESEARCH_ATTEMPTS,
        ["date", "result", "name", "shape_scope", "headline", "why_interesting", "notes"],
    )


def parse_timeline_heading(text):
    match = re.match(
        r"^(\d{4}-\d{2}-\d{2})(?: to (\d{4}-\d{2}-\d{2}))?(?:\s+([^:]+))?:\s*(.+)$",
        text,
    )
    if not match:
        return "", "", "", text
    date, date_end, context, title = match.groups()
    return date, date_end or "", context or "", title


def classify_timeline_entry(title, body):
    title_l = title.lower()
    body_l = body.lower()
    if "blocked" in title_l:
        return "blocked"
    if "accepted" in title_l and "rejected" in title_l:
        return "mixed"
    if "accepted" in title_l:
        return "accepted"
    if "rejected" in title_l:
        return "rejected"
    if "comparison" in title_l or "competitor" in title_l or "mpgemm" in title_l:
        return "comparison"
    if any(word in title_l for word in ["added", "workflow", "reporting", "harness", "diagnostics", "setup"]):
        return "tooling"
    if any(word in title_l for word in ["benchmark", "audit", "counter", "profiler", "calibration", "refreshed"]):
        return "measurement"
    has_accept = "accepted" in body_l or "source now" in body_l
    has_reject = "rejected" in body_l or "regressed" in body_l or "regression" in body_l
    if has_accept and has_reject:
        return "mixed"
    if has_accept:
        return "accepted"
    if has_reject:
        return "rejected"
    return "note"


def compact_markdown(markdown, max_chars):
    pieces = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            if pieces:
                break
            continue
        if stripped.startswith("```"):
            continue
        pieces.append(stripped.lstrip("- "))
    text = " ".join(pieces)
    return text[: max_chars - 3] + "..." if len(text) > max_chars else text


def timeline_metrics_snippet(markdown, max_chars=900):
    keywords = [
        "median",
        "bootstrap",
        "GF/s",
        "GB/s",
        "Accelerate",
        "MpGEMM",
        "passed",
        "rejected",
        "regressed",
        "neutral",
        "hard regression",
    ]
    lines = []
    for line in markdown.splitlines():
        stripped = line.strip().lstrip("- ")
        if not stripped:
            continue
        if any(keyword in stripped for keyword in keywords):
            lines.append(stripped)
        if len(lines) >= 5:
            break
    text = " ".join(lines)
    return text[: max_chars - 3] + "..." if len(text) > max_chars else text


def extract_shape_scope(text, max_shapes=20):
    shapes = []
    seen = set()
    for shape in re.findall(r"\b\d+x\d+x\d+\b", text):
        if shape in seen:
            continue
        seen.add(shape)
        shapes.append(shape)
        if len(shapes) >= max_shapes:
            break
    return ";".join(shapes)


def export_optimization_timeline():
    if not TIMELINE_PATH.exists():
        write_csv(
            DATA_DIR / "optimization_timeline.csv",
            [],
            [
                "entry_index",
                "date",
                "date_end",
                "context",
                "result",
                "title",
                "shape_scope",
                "summary",
                "metrics_snippet",
                "mentions_accepted",
                "mentions_rejected",
                "source_line",
                "details_markdown",
            ],
        )
        return

    text = TIMELINE_PATH.read_text()
    (DATA_DIR / "optimization_timeline.md").write_text(text)
    rows = []
    current = None
    body_lines = []

    def flush():
        if current is None:
            return
        body = "\n".join(body_lines).strip()
        combined = f"{current['title']}\n{body}"
        body_l = body.lower()
        row = {
            **current,
            "result": classify_timeline_entry(current["title"], body),
            "shape_scope": extract_shape_scope(combined),
            "summary": compact_markdown(body, 520),
            "metrics_snippet": timeline_metrics_snippet(body),
            "mentions_accepted": "accepted" in body_l,
            "mentions_rejected": "rejected" in body_l or "regressed" in body_l,
            "details_markdown": body,
        }
        rows.append(row)

    for line_number, line in enumerate(text.splitlines(), start=1):
        heading = re.match(r"^###\s+(.+)$", line)
        if heading:
            flush()
            date, date_end, context, title = parse_timeline_heading(heading.group(1))
            current = {
                "entry_index": len(rows) + 1,
                "date": date,
                "date_end": date_end,
                "context": context,
                "title": title,
                "source_line": line_number,
            }
            body_lines = []
        elif current is not None:
            body_lines.append(line)
    flush()

    write_csv(
        DATA_DIR / "optimization_timeline.csv",
        rows,
        [
            "entry_index",
            "date",
            "date_end",
            "context",
            "result",
            "title",
            "shape_scope",
            "summary",
            "metrics_snippet",
            "mentions_accepted",
            "mentions_rejected",
            "source_line",
            "details_markdown",
        ],
    )


def export_mpgemm():
    candidates = [
        artifact_path("cob-mpgemm-calibration-20260527-allcsv-toolcheck/mpgemm-row-sgemm.all.csv"),
        artifact_path("cob-mpgemm-calibration-20260527-full-r101-i4/mpgemm-row-sgemm.all.csv"),
    ]
    source = next((path for path in candidates if path.exists()), None)
    rows = []
    fieldnames = None
    if source is not None:
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = ["source_path"] + (reader.fieldnames or [])
            for row in reader:
                out = {"source_path": portable_source_path(source)}
                out.update(row)
                rows.append(out)
    target = DATA_DIR / "mpgemm_comparison.csv"
    if fieldnames is None and target.exists():
        return
    if fieldnames is None:
        fieldnames = ["source_path", "note"]
        rows.append({"source_path": "", "note": "No MpGEMM all-row CSV found locally."})
    write_csv(target, rows, fieldnames)


def export_metadata():
    hardware = {
        "machine": "Apple M5 Max",
        "single_thread_assumption": "one P-core in one P-cluster",
        "p_cluster_l2": "16 MB",
        "l1d": "64 KB",
        "page_size": "16 KB",
        "cache_line": "128 B",
        "sme": "SME2.1 available",
        "date": "2026-05-27",
    }
    schema = {
        "benchmark_results.csv": "Route-aware benchmark rows from selected May 2026 sweeps.",
        "paired_confirmations.csv": "Manual extraction of accepted/rejected paired confirmations from the optimization session.",
        "fresh_paired_runs.csv": "Parsed fresh repeat-61 paired A/B logs for current-vs-head and Accelerate-vs-COB.",
        "validation_runs.csv": "Fresh correctness/test command results.",
        "research_attempts.csv": "Accepted, mixed, and rejected optimization attempts worth preserving.",
        "optimization_timeline.csv": "Structured parse of docs/optimization-timeline.md with one row per dated entry.",
        "optimization_timeline.md": "Raw optimization timeline markdown copied from the source repo.",
        "mpgemm_comparison.csv": "MpGEMM FP32 row_sgemm calibration comparison when locally available.",
        "hardware_context.json": "Hardware context for the local benchmark machine.",
    }
    (DATA_DIR / "hardware_context.json").write_text(json.dumps(hardware, indent=2) + "\n")
    (DATA_DIR / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")


def sync_space_data():
    target = SPACE_DIR / "data"
    target.mkdir(parents=True, exist_ok=True)
    for source in DATA_DIR.iterdir():
        if source.is_file():
            (target / source.name).write_bytes(source.read_bytes())


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f64(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def export_static_space_index():
    benchmark_rows = read_csv_rows(DATA_DIR / "benchmark_results.csv")
    fresh_paired_rows = read_csv_rows(DATA_DIR / "fresh_paired_runs.csv")
    paired_rows = read_csv_rows(DATA_DIR / "paired_confirmations.csv")
    research_rows = read_csv_rows(DATA_DIR / "research_attempts.csv")
    timeline_rows = read_csv_rows(DATA_DIR / "optimization_timeline.csv")
    validation_rows = read_csv_rows(DATA_DIR / "validation_runs.csv")

    payload = {
        "benchmarks": [
            {
                "run_id": row.get("run_id", ""),
                "shape": row.get("shape", ""),
                "implementation": row.get("implementation", ""),
                "route": row.get("route", ""),
                "median": row.get("median_throughput", ""),
                "best": row.get("best_throughput", ""),
                "unit": row.get("unit", ""),
                "notes": row.get("notes", ""),
            }
            for row in benchmark_rows
        ],
        "freshPaired": [
            {
                "run_id": row.get("run_id", ""),
                "shape": row.get("shape", ""),
                "comparison": row.get("comparison", ""),
                "route": row.get("route", ""),
                "ratio_label": row.get("ratio_label", ""),
                "median_ratio": row.get("median_ratio", ""),
                "bootstrap_low": row.get("bootstrap_low", ""),
                "bootstrap_high": row.get("bootstrap_high", ""),
                "holdout_median": row.get("holdout_median", ""),
                "a_label": row.get("a_label", ""),
                "a_median_gflops": row.get("a_median_gflops", ""),
                "b_label": row.get("b_label", ""),
                "b_median_gflops": row.get("b_median_gflops", ""),
                "quality_note": row.get("quality_note", ""),
            }
            for row in fresh_paired_rows
        ],
        "pairedConfirmations": [
            {
                "run_id": row.get("run_id", ""),
                "shape": row.get("shape", ""),
                "comparison": row.get("comparison", ""),
                "route": row.get("route", ""),
                "ratio_label": row.get("ratio_label", ""),
                "median_ratio": row.get("median_ratio", ""),
                "bootstrap_low": row.get("bootstrap_low", ""),
                "bootstrap_high": row.get("bootstrap_high", ""),
                "holdout_median": row.get("holdout_median", ""),
                "notes": row.get("notes", ""),
            }
            for row in paired_rows
        ],
        "research": research_rows,
        "timeline": [
            {
                "entry_index": row.get("entry_index", ""),
                "date": row.get("date", ""),
                "context": row.get("context", ""),
                "result": row.get("result", ""),
                "title": row.get("title", ""),
                "shape_scope": row.get("shape_scope", ""),
                "summary": row.get("summary", ""),
                "metrics_snippet": row.get("metrics_snippet", ""),
                "source_line": row.get("source_line", ""),
            }
            for row in timeline_rows
        ],
        "validation": validation_rows,
        "counts": {
            "benchmarkRows": len(benchmark_rows),
            "freshPairedRows": len(fresh_paired_rows),
            "pairedConfirmationRows": len(paired_rows),
            "researchRows": len(research_rows),
            "timelineRows": len(timeline_rows),
        },
    }

    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    template = (ROOT / "tools" / "hf_dashboard_template.html").read_text()
    (SPACE_DIR / "index.html").write_text(template.replace("__DATA_JSON__", data_json))
    return

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>COB SGEMM Benchmark Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #080b12;
      --panel: #111723;
      --panel2: #151d2b;
      --text: #eef3ff;
      --muted: #a8b3c7;
      --line: #273247;
      --accent: #70e0b5;
      --warn: #f3c96b;
      --bad: #ff8d8d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    p {{ margin: 0; color: var(--muted); }}
    .hero {{ display: grid; gap: 18px; margin-bottom: 22px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    .metric strong {{ display: block; font-size: 23px; color: var(--accent); }}
    .metric span {{ color: var(--muted); }}
    nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0; }}
    button, select, input {{
      color: var(--text);
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 11px;
      font: inherit;
    }}
    button {{ cursor: pointer; }}
    button.active {{ border-color: var(--accent); color: var(--accent); }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 18px;
    }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 900px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #0e1420; text-align: left; color: #dfe8fb; }}
    td {{ color: #d5deef; }}
    .muted {{ color: var(--muted); }}
    .good {{ color: var(--accent); }}
    .bad {{ color: var(--bad); }}
    .warn {{ color: var(--warn); }}
    .hidden {{ display: none; }}
    footer {{ color: var(--muted); margin: 20px 0; }}
    a {{ color: #9bd3ff; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div>
        <h1>COB SGEMM Benchmark Dashboard</h1>
        <p>Static dashboard for Apple Silicon single-thread FP32 row-major SGEMM evidence. It renders directly from the exported dataset files, with no Gradio runtime.</p>
      </div>
      <div class="summary" id="summary"></div>
    </section>

    <nav>
      <button class="active" data-tab="overview">Overview</button>
      <button data-tab="benchmarks">Benchmarks</button>
      <button data-tab="fresh">Fresh paired runs</button>
      <button data-tab="timeline">Timeline</button>
      <button data-tab="research">Research attempts</button>
    </nav>

    <section id="overview" class="tab"></section>
    <section id="benchmarks" class="tab hidden"></section>
    <section id="fresh" class="tab hidden"></section>
    <section id="timeline" class="tab hidden"></section>
    <section id="research" class="tab hidden"></section>

    <footer>
      Scope: CPU-only, single-thread FP32 row-major SGEMM on Apple Silicon. Apple Accelerate is tracked separately because it is proprietary.
    </footer>
  </main>

  <script id="data" type="application/json">{data_json}</script>
  <script>
    const DATA = JSON.parse(document.getElementById("data").textContent);
    const fmt = (value, digits = 3) => {{
      const number = Number(value);
      return Number.isFinite(number) ? number.toFixed(digits) : (value || "");
    }};
    const includes = (row, query) => !query || Object.values(row).join(" ").toLowerCase().includes(query.toLowerCase());
    const el = (id) => document.getElementById(id);

    function table(rows, columns, limit = 100) {{
      const shown = rows.slice(0, limit);
      const head = columns.map(([key, label]) => `<th>${{label}}</th>`).join("");
      const body = shown.map(row => `<tr>${{columns.map(([key]) => `<td>${{row[key] ?? ""}}</td>`).join("")}}</tr>`).join("");
      return `<div class="table-wrap"><table><thead><tr>${{head}}</tr></thead><tbody>${{body}}</tbody></table></div>`;
    }}

    function renderSummary() {{
      const items = [
        ["Benchmark rows", DATA.counts.benchmarkRows],
        ["Fresh paired rows", DATA.counts.freshPairedRows],
        ["Timeline entries", DATA.counts.timelineRows],
        ["Research rows", DATA.counts.researchRows],
        ["Validation", DATA.validation[0]?.reported_shapes ? `${{DATA.validation[0].reported_shapes}} shapes` : "loaded"],
      ];
      el("summary").innerHTML = items.map(([label, value]) => `<div class="metric"><strong>${{value}}</strong><span>${{label}}</span></div>`).join("");
    }}

    function renderOverview() {{
      const packed = DATA.benchmarks
        .filter(row => row.implementation === "cob packed-AB")
        .sort((a, b) => Number(b.median) - Number(a.median))
        .slice(0, 8)
        .map(row => ({{...row, median: fmt(row.median), best: fmt(row.best)}}));
      const sourceWins = DATA.freshPaired
        .filter(row => row.ratio_label === "current/repo_head" && Number(row.median_ratio) > 1)
        .sort((a, b) => Number(b.median_ratio) - Number(a.median_ratio))
        .map(row => ({{...row, median_ratio: fmt(row.median_ratio, 4), bootstrap: `[${{fmt(row.bootstrap_low, 4)}}, ${{fmt(row.bootstrap_high, 4)}}]`}}));
      const accelerateGaps = DATA.freshPaired
        .filter(row => row.ratio_label === "Accelerate/COB" && Number(row.median_ratio) > 1)
        .sort((a, b) => Number(b.median_ratio) - Number(a.median_ratio))
        .map(row => ({{...row, median_ratio: fmt(row.median_ratio, 4), bootstrap: `[${{fmt(row.bootstrap_low, 4)}}, ${{fmt(row.bootstrap_high, 4)}}]`}}));
      el("overview").innerHTML = `
        <div class="panel">
          <h2>Highlights</h2>
          <p>The most interesting additions are fresh paired A/B runs, packed-AB headline rows near 1.9 TFLOP/s median, and the 512x1216x3072 source-route win. The less flattering Accelerate gaps and rejected probes are included too.</p>
        </div>
        <div class="panel">
          <h2>Top packed-AB throughput</h2>
          ${{table(packed, [["shape","shape"],["median","median GF/s"],["best","best GF/s"],["route","route"],["run_id","run"]], 8)}}
        </div>
        <div class="panel">
          <h2>Fresh current-vs-head wins</h2>
          ${{table(sourceWins, [["shape","shape"],["median_ratio","current/head"],["bootstrap","bootstrap"],["route","route"],["quality_note","quality"]], 8)}}
        </div>
        <div class="panel">
          <h2>Fresh direct Accelerate gaps</h2>
          ${{table(accelerateGaps, [["shape","shape"],["median_ratio","Accelerate/COB"],["bootstrap","bootstrap"],["route","route"],["quality_note","quality"]], 10)}}
        </div>
      `;
    }}

    function renderBenchmarks(query = "") {{
      const rows = DATA.benchmarks
        .filter(row => includes(row, query))
        .sort((a, b) => Number(b.median) - Number(a.median))
        .map(row => ({{...row, median: fmt(row.median), best: fmt(row.best)}}));
      el("benchmarks").innerHTML = `
        <div class="panel">
          <h2>Benchmark rows</h2>
          <div class="controls"><input id="benchQuery" placeholder="Search shape, route, run" value="${{query}}"></div>
          ${{table(rows, [["shape","shape"],["implementation","implementation"],["median","median"],["best","best"],["unit","unit"],["route","route"],["run_id","run"]], 160)}}
        </div>
      `;
      document.getElementById("benchQuery").addEventListener("input", (event) => renderBenchmarks(event.target.value));
    }}

    function renderFresh(query = "") {{
      const rows = DATA.freshPaired
        .filter(row => includes(row, query))
        .map(row => ({{...row, median_ratio: fmt(row.median_ratio, 4), bootstrap: `[${{fmt(row.bootstrap_low, 4)}}, ${{fmt(row.bootstrap_high, 4)}}]`}}));
      el("fresh").innerHTML = `
        <div class="panel">
          <h2>Fresh paired runs</h2>
          <div class="controls"><input id="freshQuery" placeholder="Search shape, comparison, route" value="${{query}}"></div>
          ${{table(rows, [["shape","shape"],["comparison","comparison"],["ratio_label","ratio"],["median_ratio","median"],["bootstrap","bootstrap"],["a_median_gflops","A GF/s"],["b_median_gflops","B GF/s"],["quality_note","quality"]], 160)}}
        </div>
      `;
      document.getElementById("freshQuery").addEventListener("input", (event) => renderFresh(event.target.value));
    }}

    function renderTimeline(result = "All", query = "") {{
      const results = ["All", ...Array.from(new Set(DATA.timeline.map(row => row.result))).sort()];
      const rows = DATA.timeline
        .filter(row => result === "All" || row.result === result)
        .filter(row => includes(row, query));
      el("timeline").innerHTML = `
        <div class="panel">
          <h2>Optimization timeline</h2>
          <div class="controls">
            <select id="timelineResult">${{results.map(value => `<option ${{value === result ? "selected" : ""}}>${{value}}</option>`).join("")}}</select>
            <input id="timelineQuery" placeholder="Search title, shape, summary" value="${{query}}">
          </div>
          ${{table(rows, [["entry_index","entry"],["date","date"],["result","result"],["title","title"],["shape_scope","shape scope"],["summary","summary"],["metrics_snippet","metrics"]], 190)}}
        </div>
      `;
      document.getElementById("timelineResult").addEventListener("change", (event) => renderTimeline(event.target.value, query));
      document.getElementById("timelineQuery").addEventListener("input", (event) => renderTimeline(result, event.target.value));
    }}

    function renderResearch(result = "All") {{
      const results = ["All", ...Array.from(new Set(DATA.research.map(row => row.result))).sort()];
      const rows = DATA.research.filter(row => result === "All" || row.result === result);
      el("research").innerHTML = `
        <div class="panel">
          <h2>Research attempts</h2>
          <div class="controls"><select id="researchResult">${{results.map(value => `<option ${{value === result ? "selected" : ""}}>${{value}}</option>`).join("")}}</select></div>
          ${{table(rows, [["date","date"],["result","result"],["name","name"],["shape_scope","shape scope"],["headline","headline"],["why_interesting","why interesting"],["notes","notes"]], 100)}}
        </div>
      `;
      document.getElementById("researchResult").addEventListener("change", (event) => renderResearch(event.target.value));
    }}

    function setTab(id) {{
      document.querySelectorAll(".tab").forEach(node => node.classList.add("hidden"));
      document.querySelectorAll("nav button").forEach(node => node.classList.toggle("active", node.dataset.tab === id));
      el(id).classList.remove("hidden");
    }}

    document.querySelectorAll("nav button").forEach(button => button.addEventListener("click", () => setTab(button.dataset.tab)));
    renderSummary();
    renderOverview();
    renderBenchmarks();
    renderFresh();
    renderTimeline();
    renderResearch();
  </script>
</body>
</html>
"""
    (SPACE_DIR / "index.html").write_text(html)


def main():
    export_benchmarks()
    export_paired_confirmations()
    export_fresh_paired_runs()
    export_validation_runs()
    export_research_attempts()
    export_optimization_timeline()
    export_mpgemm()
    export_metadata()
    sync_space_data()
    export_static_space_index()


if __name__ == "__main__":
    main()
