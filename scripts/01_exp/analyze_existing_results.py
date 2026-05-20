import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASELINE_SUMMARY = Path("outputs/tables/baseline_summary.csv")
SC_CISC_SUMMARY = Path("outputs/tables/sc_cisc_summary.csv")

BASELINE_PARSED_FILES = [
    Path("outputs/parsed/baseline/gsm8k_platinum_direct_parsed.jsonl"),
    Path("outputs/parsed/baseline/gsm8k_platinum_cot_parsed.jsonl"),
    Path("outputs/parsed/baseline/math500_direct_parsed.jsonl"),
    Path("outputs/parsed/baseline/math500_cot_parsed.jsonl"),
]

SC_CISC_QUESTION_FILES = [
    Path("outputs/parsed/sc_cisc/gsm8k_platinum_sc_k5_questions.jsonl"),
    Path("outputs/parsed/sc_cisc/gsm8k_platinum_cisc_k5_questions.jsonl"),
    Path("outputs/parsed/sc_cisc/math500_sc_k5_questions.jsonl"),
    Path("outputs/parsed/sc_cisc/math500_cisc_k5_questions.jsonl"),
]

CISC_CANDIDATE_FILES = [
    Path("outputs/parsed/sc_cisc/gsm8k_platinum_cisc_k5_candidates_parsed.jsonl"),
    Path("outputs/parsed/sc_cisc/math500_cisc_k5_candidates_parsed.jsonl"),
]

OUT_ANALYSIS = Path("outputs/analysis")
OUT_FIGURES = Path("outputs/figures")
OUT_ERROR_CASES = OUT_ANALYSIS / "error_cases"

METHOD_ORDER = ["direct", "cot", "sc", "cisc"]
METHOD_LABEL = {"direct": "Direct@1", "cot": "CoT@1", "sc": "SC@5", "cisc": "CISC@5"}
METHOD_COLOR = {
    "direct": (45, 114, 143),
    "cot": (231, 111, 81),
    "sc": (42, 157, 143),
    "cisc": (233, 196, 106),
}

ERROR_CASE_DEFS = [
    ("direct_wrong_cot_right", "Direct错CoT对", "direct", False, "cot", True),
    ("cot_wrong_sc_right", "CoT错SC对", "cot", False, "sc", True),
    ("cot_right_sc_wrong", "CoT对SC错", "cot", True, "sc", False),
    ("sc_wrong_cisc_right", "SC错CISC对", "sc", False, "cisc", True),
    ("sc_right_cisc_wrong", "SC对CISC错", "sc", True, "cisc", False),
]


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def quantile(values, q):
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1 - frac) + vals[hi] * frac)


def summary_stats(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    vals = [float(v) for v in values]
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "median": quantile(vals, 0.5),
        "p10": quantile(vals, 0.1),
        "p25": quantile(vals, 0.25),
        "p75": quantile(vals, 0.75),
        "p90": quantile(vals, 0.9),
        "min": min(vals),
        "max": max(vals),
    }


def method_record_from_baseline_row(row):
    return {
        "dataset": row.get("dataset"),
        "id": row.get("id"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "pred_answer": row.get("pred_answer"),
        "is_correct": bool(row.get("is_correct")),
        "needs_review": bool(row.get("needs_review")),
        "parse_failed": row.get("pred_answer") is None,
        "output_tokens": to_float(row.get("output_tokens"), default=0.0),
        "latency_seconds": to_float(row.get("latency_seconds"), default=0.0),
        "vote_details": None,
    }


def method_record_from_sc_cisc_row(row):
    return {
        "dataset": row.get("dataset"),
        "id": row.get("id"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "pred_answer": row.get("final_answer"),
        "is_correct": bool(row.get("is_correct")),
        "needs_review": bool(row.get("needs_review")),
        "parse_failed": bool(row.get("all_parse_failed")),
        "output_tokens": to_float(row.get("total_output_tokens"), default=0.0),
        "latency_seconds": to_float(row.get("total_latency_seconds"), default=0.0),
        "vote_details": row.get("vote_details"),
    }


def load_all_method_records():
    by_key = defaultdict(dict)

    for path in BASELINE_PARSED_FILES:
        rows = read_jsonl(path)
        for row in rows:
            key = (row.get("dataset"), row.get("id"))
            by_key[key][row.get("method")] = method_record_from_baseline_row(row)

    for path in SC_CISC_QUESTION_FILES:
        rows = read_jsonl(path)
        for row in rows:
            key = (row.get("dataset"), row.get("id"))
            by_key[key][row.get("method")] = method_record_from_sc_cisc_row(row)

    return by_key


def merge_master_summary():
    baseline_rows = read_csv(BASELINE_SUMMARY)
    sc_rows = read_csv(SC_CISC_SUMMARY)
    merged = []

    for row in baseline_rows:
        dataset = row["dataset"]
        method = row["method"]
        n = to_int(row.get("n"))
        avg_tokens = to_float(row.get("avg_output_tokens"))
        avg_latency = to_float(row.get("avg_latency_seconds"))
        merged.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABEL.get(method, method),
                "k": 1,
                "n": n,
                "correct": to_int(row.get("correct")),
                "accuracy": to_float(row.get("accuracy")),
                "parse_failed": to_int(row.get("parse_failed")),
                "parse_failed_rate": to_float(row.get("parse_failed_rate")),
                "needs_review": to_int(row.get("needs_review")),
                "needs_review_rate": to_float(row.get("needs_review_rate")),
                "hit_max": to_int(row.get("hit_max")),
                "hit_max_rate": to_float(row.get("hit_max_rate")),
                "avg_output_tokens_per_question": avg_tokens,
                "avg_latency_seconds_per_question": avg_latency,
                "avg_output_tokens_per_candidate": avg_tokens,
                "avg_latency_seconds_per_candidate": avg_latency,
                "source_table": str(BASELINE_SUMMARY),
            }
        )

    for row in sc_rows:
        dataset = row["dataset"]
        method = row["method"]
        merged.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABEL.get(method, method),
                "k": to_int(row.get("k")),
                "n": to_int(row.get("n")),
                "correct": to_int(row.get("correct")),
                "accuracy": to_float(row.get("accuracy")),
                "parse_failed": to_int(row.get("all_parse_failed")),
                "parse_failed_rate": to_float(row.get("all_parse_failed_rate")),
                "needs_review": to_int(row.get("needs_review")),
                "needs_review_rate": to_float(row.get("needs_review_rate")),
                "hit_max": None,
                "hit_max_rate": None,
                "avg_output_tokens_per_question": to_float(
                    row.get("avg_total_output_tokens_per_question")
                ),
                "avg_latency_seconds_per_question": to_float(
                    row.get("avg_total_latency_seconds_per_question")
                ),
                "avg_output_tokens_per_candidate": to_float(
                    row.get("avg_output_tokens_per_candidate")
                ),
                "avg_latency_seconds_per_candidate": to_float(
                    row.get("avg_latency_seconds_per_candidate")
                ),
                "source_table": str(SC_CISC_SUMMARY),
            }
        )

    merged.sort(key=lambda r: (r["dataset"], METHOD_ORDER.index(r["method"])))
    write_csv(
        OUT_ANALYSIS / "master_results.csv",
        merged,
        [
            "dataset",
            "method",
            "method_label",
            "k",
            "n",
            "correct",
            "accuracy",
            "parse_failed",
            "parse_failed_rate",
            "needs_review",
            "needs_review_rate",
            "hit_max",
            "hit_max_rate",
            "avg_output_tokens_per_question",
            "avg_latency_seconds_per_question",
            "avg_output_tokens_per_candidate",
            "avg_latency_seconds_per_candidate",
            "source_table",
        ],
    )
    return merged


def load_cisc_candidate_confidence_by_question():
    question_conf = defaultdict(list)
    all_rows = []
    for path in CISC_CANDIDATE_FILES:
        for row in read_jsonl(path):
            key = (row.get("dataset"), row.get("id"))
            question_conf[key].append(row)
            all_rows.append(row)
    return question_conf, all_rows


def question_confidence_summary(candidates):
    confs = []
    wrong_confs = []
    right_confs = []
    missing = 0
    for row in candidates:
        conf = row.get("confidence")
        if conf is None:
            missing += 1
            continue
        conf = float(conf)
        confs.append(conf)
        if row.get("candidate_is_correct"):
            right_confs.append(conf)
        else:
            wrong_confs.append(conf)
    return {
        "cisc_candidate_count": len(candidates),
        "cisc_missing_confidence_count": missing,
        "cisc_conf_min": min(confs) if confs else None,
        "cisc_conf_max": max(confs) if confs else None,
        "cisc_conf_mean": sum(confs) / len(confs) if confs else None,
        "cisc_wrong_candidate_count": len(wrong_confs),
        "cisc_wrong_conf_max": max(wrong_confs) if wrong_confs else None,
        "cisc_wrong_conf_mean": (sum(wrong_confs) / len(wrong_confs)) if wrong_confs else None,
        "cisc_right_candidate_count": len(right_confs),
        "cisc_right_conf_mean": (sum(right_confs) / len(right_confs)) if right_confs else None,
    }


def export_error_cases(method_records, cisc_conf_by_question):
    counts_rows = []
    for filename, label, left_method, left_correct, right_method, right_correct in ERROR_CASE_DEFS:
        case_rows = []
        by_dataset_counts = defaultdict(int)
        for key, methods in method_records.items():
            left = methods.get(left_method)
            right = methods.get(right_method)
            if left is None or right is None:
                continue
            if bool(left.get("is_correct")) != bool(left_correct):
                continue
            if bool(right.get("is_correct")) != bool(right_correct):
                continue

            dataset, qid = key
            cisc_extra = question_confidence_summary(cisc_conf_by_question.get(key, []))
            row = {
                "case_type": label,
                "dataset": dataset,
                "id": qid,
                "question": (methods.get("cot") or methods.get("direct") or methods.get("sc") or methods.get("cisc") or {}).get("question"),
                "gold_answer": (methods.get("cot") or methods.get("direct") or methods.get("sc") or methods.get("cisc") or {}).get("gold_answer"),
                "direct_pred": (methods.get("direct") or {}).get("pred_answer"),
                "direct_is_correct": (methods.get("direct") or {}).get("is_correct"),
                "cot_pred": (methods.get("cot") or {}).get("pred_answer"),
                "cot_is_correct": (methods.get("cot") or {}).get("is_correct"),
                "sc_pred": (methods.get("sc") or {}).get("pred_answer"),
                "sc_is_correct": (methods.get("sc") or {}).get("is_correct"),
                "sc_vote_details": json.dumps((methods.get("sc") or {}).get("vote_details"), ensure_ascii=False),
                "cisc_pred": (methods.get("cisc") or {}).get("pred_answer"),
                "cisc_is_correct": (methods.get("cisc") or {}).get("is_correct"),
                "cisc_vote_details": json.dumps((methods.get("cisc") or {}).get("vote_details"), ensure_ascii=False),
            }
            row.update(cisc_extra)
            case_rows.append(row)
            by_dataset_counts[dataset] += 1

        case_rows.sort(key=lambda r: (r["dataset"], r["id"]))
        fieldnames = list(case_rows[0].keys()) if case_rows else [
            "case_type",
            "dataset",
            "id",
            "question",
            "gold_answer",
            "direct_pred",
            "direct_is_correct",
            "cot_pred",
            "cot_is_correct",
            "sc_pred",
            "sc_is_correct",
            "sc_vote_details",
            "cisc_pred",
            "cisc_is_correct",
            "cisc_vote_details",
        ]

        write_csv(OUT_ERROR_CASES / f"{filename}.csv", case_rows, fieldnames)
        write_jsonl(OUT_ERROR_CASES / f"{filename}.jsonl", case_rows)

        total = len(case_rows)
        counts_rows.append({"case_type": label, "dataset": "all", "count": total})
        for dataset, count in sorted(by_dataset_counts.items()):
            counts_rows.append({"case_type": label, "dataset": dataset, "count": count})

    write_csv(
        OUT_ERROR_CASES / "error_case_counts.csv",
        counts_rows,
        ["case_type", "dataset", "count"],
    )


def analyze_cisc_confidence(candidates):
    by_dataset = defaultdict(list)
    for row in candidates:
        by_dataset[row.get("dataset")].append(row)
    by_dataset["all"] = list(candidates)

    dist_rows = []
    high_conf_rows = []
    quality_rows = []
    candidate_export = []
    thresholds = [70, 80, 90, 95, 100]

    for dataset, rows in sorted(by_dataset.items()):
        correct_confs = []
        wrong_confs = []
        valid_count = 0
        missing = 0
        parse_failed = 0

        for row in rows:
            conf = row.get("confidence")
            is_parse_failed = bool(row.get("candidate_parse_failed"))
            if is_parse_failed:
                parse_failed += 1

            if conf is None:
                missing += 1
                continue
            conf = float(conf)
            valid_count += 1

            candidate_export.append(
                {
                    "dataset": row.get("dataset"),
                    "id": row.get("id"),
                    "trial_id": row.get("trial_id"),
                    "confidence": conf,
                    "candidate_is_correct": bool(row.get("candidate_is_correct")),
                    "candidate_parse_failed": is_parse_failed,
                    "pred_answer": row.get("pred_answer"),
                    "gold_answer": row.get("gold_answer"),
                }
            )

            if row.get("candidate_is_correct"):
                correct_confs.append(conf)
            else:
                wrong_confs.append(conf)

        quality_rows.append(
            {
                "dataset": dataset,
                "total_candidates": len(rows),
                "valid_confidence_candidates": valid_count,
                "missing_confidence_candidates": missing,
                "missing_confidence_rate": (missing / len(rows)) if rows else None,
                "candidate_parse_failed": parse_failed,
                "candidate_parse_failed_rate": (parse_failed / len(rows)) if rows else None,
            }
        )

        for group_name, values in [("correct", correct_confs), ("incorrect", wrong_confs)]:
            stats = summary_stats(values)
            dist_rows.append(
                {
                    "dataset": dataset,
                    "group": group_name,
                    "count": stats["count"],
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "p10": stats["p10"],
                    "p25": stats["p25"],
                    "p75": stats["p75"],
                    "p90": stats["p90"],
                    "min": stats["min"],
                    "max": stats["max"],
                }
            )

        wrong_n = len(wrong_confs)
        for thr in thresholds:
            high_wrong = sum(1 for c in wrong_confs if c >= thr)
            high_conf_rows.append(
                {
                    "dataset": dataset,
                    "threshold": thr,
                    "wrong_candidate_count": wrong_n,
                    "high_conf_wrong_count": high_wrong,
                    "high_conf_error_rate_among_errors": (high_wrong / wrong_n) if wrong_n else None,
                    "high_conf_error_rate_overall_valid": (high_wrong / valid_count) if valid_count else None,
                }
            )

    write_csv(
        OUT_ANALYSIS / "cisc_confidence_distribution_stats.csv",
        dist_rows,
        ["dataset", "group", "count", "mean", "median", "p10", "p25", "p75", "p90", "min", "max"],
    )
    write_csv(
        OUT_ANALYSIS / "cisc_high_conf_error_rates.csv",
        high_conf_rows,
        [
            "dataset",
            "threshold",
            "wrong_candidate_count",
            "high_conf_wrong_count",
            "high_conf_error_rate_among_errors",
            "high_conf_error_rate_overall_valid",
        ],
    )
    write_csv(
        OUT_ANALYSIS / "cisc_confidence_data_quality.csv",
        quality_rows,
        [
            "dataset",
            "total_candidates",
            "valid_confidence_candidates",
            "missing_confidence_candidates",
            "missing_confidence_rate",
            "candidate_parse_failed",
            "candidate_parse_failed_rate",
        ],
    )
    write_csv(
        OUT_ANALYSIS / "cisc_candidates_confidence_records.csv",
        candidate_export,
        [
            "dataset",
            "id",
            "trial_id",
            "confidence",
            "candidate_is_correct",
            "candidate_parse_failed",
            "pred_answer",
            "gold_answer",
        ],
    )

    return candidate_export


def load_font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_axis(draw, x0, y0, x1, y1, color=(20, 20, 20)):
    draw.line([(x0, y0), (x0, y1)], fill=color, width=2)
    draw.line([(x0, y1), (x1, y1)], fill=color, width=2)


def draw_grouped_bar_chart(summary_rows, metric_key, title, y_label, output_path):
    datasets = sorted(set(r["dataset"] for r in summary_rows))
    width, height = 1300, 760
    margin_left, margin_right = 90, 50
    margin_top, margin_bottom = 100, 130
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    img = Image.new("RGB", (width, height), (248, 249, 250))
    draw = ImageDraw.Draw(img)
    font_title = load_font(30)
    font_axis = load_font(18)
    font_small = load_font(14)

    max_val = 0.0
    for row in summary_rows:
        max_val = max(max_val, float(row.get(metric_key, 0.0)))
    if max_val <= 0:
        max_val = 1.0
    y_max = max_val * 1.15

    # Background grid
    ticks = 5
    for i in range(ticks + 1):
        y = margin_top + int(chart_h * i / ticks)
        draw.line([(margin_left, y), (margin_left + chart_w, y)], fill=(220, 224, 229), width=1)
        v = y_max * (1 - i / ticks)
        draw.text((15, y - 8), f"{v:.2f}", fill=(70, 70, 70), font=font_small)

    _draw_axis(draw, margin_left, margin_top, margin_left + chart_w, margin_top + chart_h)
    draw.text((width // 2 - 240, 24), title, fill=(20, 20, 20), font=font_title)
    draw.text((20, margin_top - 40), y_label, fill=(50, 50, 50), font=font_axis)

    group_count = len(datasets)
    group_w = chart_w / max(group_count, 1)
    bar_area_ratio = 0.72
    bar_gap = 6
    method_count = len(METHOD_ORDER)

    for gi, dataset in enumerate(datasets):
        group_x0 = margin_left + gi * group_w
        content_w = group_w * bar_area_ratio
        content_x0 = group_x0 + (group_w - content_w) / 2
        bar_w = (content_w - (method_count - 1) * bar_gap) / method_count

        for mi, method in enumerate(METHOD_ORDER):
            match = [r for r in summary_rows if r["dataset"] == dataset and r["method"] == method]
            if not match:
                continue
            v = float(match[0].get(metric_key, 0.0))
            h = 0 if y_max == 0 else (v / y_max) * chart_h
            x0 = int(content_x0 + mi * (bar_w + bar_gap))
            x1 = int(x0 + bar_w)
            y1 = margin_top + chart_h
            y0 = int(y1 - h)
            draw.rectangle([(x0, y0), (x1, y1)], fill=METHOD_COLOR[method], outline=(30, 30, 30), width=1)
            draw.text((x0, y0 - 18), f"{v:.3g}", fill=(30, 30, 30), font=font_small)

        label = dataset.replace("_", "\n")
        draw.multiline_text((int(group_x0 + group_w / 2 - 50), margin_top + chart_h + 10), label, fill=(40, 40, 40), font=font_axis, align="center")

    # Legend
    lx = width - 420
    ly = 30
    for i, method in enumerate(METHOD_ORDER):
        y = ly + i * 26
        draw.rectangle([(lx, y), (lx + 18, y + 18)], fill=METHOD_COLOR[method], outline=(20, 20, 20))
        draw.text((lx + 26, y), METHOD_LABEL[method], fill=(30, 30, 30), font=font_small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def draw_confidence_histogram(candidate_rows, output_path):
    # Overall bins for correct vs incorrect
    bins = [(i, i + 9) for i in range(0, 100, 10)] + [(100, 100)]
    correct_counts = [0] * len(bins)
    wrong_counts = [0] * len(bins)

    for row in candidate_rows:
        conf = row.get("confidence")
        if conf is None:
            continue
        conf = int(round(float(conf)))
        idx = min(conf // 10, 10)
        if row.get("candidate_is_correct"):
            correct_counts[idx] += 1
        else:
            wrong_counts[idx] += 1

    width, height = 1300, 760
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 100, 120
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    img = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    font_title = load_font(30)
    font_axis = load_font(18)
    font_small = load_font(14)

    max_count = max(correct_counts + wrong_counts) if (correct_counts or wrong_counts) else 1
    y_max = max(1, int(max_count * 1.2))

    ticks = 6
    for i in range(ticks + 1):
        y = margin_top + int(chart_h * i / ticks)
        draw.line([(margin_left, y), (margin_left + chart_w, y)], fill=(226, 226, 220), width=1)
        v = int(round(y_max * (1 - i / ticks)))
        draw.text((15, y - 8), f"{v}", fill=(60, 60, 60), font=font_small)

    _draw_axis(draw, margin_left, margin_top, margin_left + chart_w, margin_top + chart_h)
    draw.text((width // 2 - 270, 24), "CISC Candidate Confidence Distribution", fill=(20, 20, 20), font=font_title)

    n = len(bins)
    slot_w = chart_w / max(n, 1)
    bar_w = slot_w * 0.35

    for i in range(n):
        cx = margin_left + i * slot_w + slot_w * 0.15
        wx = margin_left + i * slot_w + slot_w * 0.55

        c_count = correct_counts[i]
        w_count = wrong_counts[i]

        c_h = (c_count / y_max) * chart_h if y_max else 0
        w_h = (w_count / y_max) * chart_h if y_max else 0

        draw.rectangle(
            [(int(cx), int(margin_top + chart_h - c_h)), (int(cx + bar_w), int(margin_top + chart_h))],
            fill=(42, 157, 143),
            outline=(20, 20, 20),
        )
        draw.rectangle(
            [(int(wx), int(margin_top + chart_h - w_h)), (int(wx + bar_w), int(margin_top + chart_h))],
            fill=(231, 111, 81),
            outline=(20, 20, 20),
        )

        label = "100" if i == 10 else f"{i*10}-{i*10+9}"
        draw.text((int(margin_left + i * slot_w + 2), margin_top + chart_h + 8), label, fill=(40, 40, 40), font=font_small)

    draw.text((margin_left + chart_w // 2 - 90, height - 45), "Confidence Bin", fill=(40, 40, 40), font=font_axis)
    draw.text((20, margin_top - 40), "Candidate Count", fill=(40, 40, 40), font=font_axis)

    lx, ly = width - 300, 32
    draw.rectangle([(lx, ly), (lx + 18, ly + 18)], fill=(42, 157, 143), outline=(20, 20, 20))
    draw.text((lx + 24, ly), "Correct candidates", fill=(30, 30, 30), font=font_small)
    draw.rectangle([(lx, ly + 26), (lx + 18, ly + 44)], fill=(231, 111, 81), outline=(20, 20, 20))
    draw.text((lx + 24, ly + 26), "Incorrect candidates", fill=(30, 30, 30), font=font_small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def generate_figures(master_rows, cisc_candidate_export):
    draw_grouped_bar_chart(
        master_rows,
        metric_key="accuracy",
        title="Accuracy Comparison Across Methods",
        y_label="Accuracy",
        output_path=OUT_FIGURES / "accuracy_comparison.png",
    )
    draw_grouped_bar_chart(
        master_rows,
        metric_key="avg_output_tokens_per_question",
        title="Average Token Cost Per Question",
        y_label="Avg Tokens",
        output_path=OUT_FIGURES / "avg_token_cost_comparison.png",
    )
    draw_grouped_bar_chart(
        master_rows,
        metric_key="avg_latency_seconds_per_question",
        title="Average Latency Cost Per Question",
        y_label="Avg Latency (s)",
        output_path=OUT_FIGURES / "avg_latency_cost_comparison.png",
    )
    draw_confidence_histogram(
        cisc_candidate_export,
        output_path=OUT_FIGURES / "cisc_confidence_distribution.png",
    )


def write_quick_report(master_rows):
    # Build compact narrative summary for convenient report copy/paste.
    best_by_dataset = {}
    for dataset in sorted(set(r["dataset"] for r in master_rows)):
        rows = [r for r in master_rows if r["dataset"] == dataset]
        best = max(rows, key=lambda r: r["accuracy"])
        best_by_dataset[dataset] = best

    lines = []
    lines.append("# Analysis Summary")
    lines.append("")
    lines.append("## Best Accuracy By Dataset")
    for dataset, row in best_by_dataset.items():
        lines.append(
            f"- {dataset}: {row['method_label']} accuracy={row['accuracy']:.4f} "
            f"(correct={row['correct']}/{row['n']})"
        )
    lines.append("")
    lines.append("## Output Files")
    lines.append("- master table: outputs/analysis/master_results.csv")
    lines.append("- error cases: outputs/analysis/error_cases/")
    lines.append("- cisc confidence stats: outputs/analysis/cisc_*")
    lines.append("- figures: outputs/figures/*.png")
    (OUT_ANALYSIS / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUT_ERROR_CASES.mkdir(parents=True, exist_ok=True)

    master_rows = merge_master_summary()
    method_records = load_all_method_records()
    cisc_conf_by_question, cisc_candidates = load_cisc_candidate_confidence_by_question()
    export_error_cases(method_records, cisc_conf_by_question)
    cisc_candidate_export = analyze_cisc_confidence(cisc_candidates)
    generate_figures(master_rows, cisc_candidate_export)
    write_quick_report(master_rows)

    print("Analysis completed.")
    print(f"Master table: {OUT_ANALYSIS / 'master_results.csv'}")
    print(f"Error cases: {OUT_ERROR_CASES}")
    print(f"Confidence analysis: {OUT_ANALYSIS}")
    print(f"Figures: {OUT_FIGURES}")


if __name__ == "__main__":
    main()
