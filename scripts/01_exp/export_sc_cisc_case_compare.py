import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_SC_QUESTION_FILES = [
    Path("outputs/parsed/sc_cisc/gsm8k_platinum_sc_k5_questions.jsonl"),
    Path("outputs/parsed/sc_cisc/math500_sc_k5_questions.jsonl"),
]

DEFAULT_CISC_QUESTION_FILES = [
    Path("outputs/parsed/sc_cisc/gsm8k_platinum_cisc_k5_questions.jsonl"),
    Path("outputs/parsed/sc_cisc/math500_cisc_k5_questions.jsonl"),
]

DEFAULT_SC_CANDIDATE_FILES = [
    Path("outputs/parsed/sc_cisc/gsm8k_platinum_sc_k5_candidates_parsed.jsonl"),
    Path("outputs/parsed/sc_cisc/math500_sc_k5_candidates_parsed.jsonl"),
]

DEFAULT_CISC_CANDIDATE_FILES = [
    Path("outputs/parsed/sc_cisc/gsm8k_platinum_cisc_k5_candidates_parsed.jsonl"),
    Path("outputs/parsed/sc_cisc/math500_cisc_k5_candidates_parsed.jsonl"),
]


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
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


def truncate_text(text, limit=600):
    if text is None:
        return None
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if len(s) <= limit:
        return s
    return s[:limit] + "...(truncated)"


def infer_problem_source(dataset):
    if dataset == "gsm8k_platinum":
        return "GSM8K-Platinum"
    if dataset == "math500":
        return "MATH-500"
    return dataset


def infer_problem_type(dataset, qid):
    if dataset == "math500" and isinstance(qid, str):
        parts = qid.split("/")
        if len(parts) >= 3:
            return parts[1]  # e.g. algebra, number_theory
        return "math500_other"
    if dataset == "gsm8k_platinum":
        return "word_problem"
    return "unknown"


def build_question_map(paths):
    question_map = {}
    for path in paths:
        for row in read_jsonl(path):
            key = (row.get("dataset"), row.get("id"))
            question_map[key] = row
    return question_map


def build_candidate_map(paths):
    candidate_map = defaultdict(list)
    for path in paths:
        for row in read_jsonl(path):
            key = (row.get("dataset"), row.get("id"))
            candidate_map[key].append(row)
    for key in candidate_map:
        candidate_map[key].sort(key=lambda r: (r.get("trial_id", -1)))
    return candidate_map


def build_10_raw_answers(sc_candidates, cisc_candidates):
    # 固定输出10条：前5条SC，后5条CISC；不足时补空
    rows = []
    for cand in sc_candidates[:5]:
        rows.append(
            {
                "method": "sc",
                "trial_id": cand.get("trial_id"),
                "pred_answer": cand.get("pred_answer"),
                "confidence": cand.get("confidence"),
                "candidate_is_correct": cand.get("candidate_is_correct"),
                "candidate_parse_failed": cand.get("candidate_parse_failed"),
                "raw_output": cand.get("raw_output"),
            }
        )
    while len(rows) < 5:
        rows.append(
            {
                "method": "sc",
                "trial_id": None,
                "pred_answer": None,
                "confidence": None,
                "candidate_is_correct": None,
                "candidate_parse_failed": None,
                "raw_output": None,
            }
        )

    cisc_rows = []
    for cand in cisc_candidates[:5]:
        cisc_rows.append(
            {
                "method": "cisc",
                "trial_id": cand.get("trial_id"),
                "pred_answer": cand.get("pred_answer"),
                "confidence": cand.get("confidence"),
                "candidate_is_correct": cand.get("candidate_is_correct"),
                "candidate_parse_failed": cand.get("candidate_parse_failed"),
                "raw_output": cand.get("raw_output"),
            }
        )
    while len(cisc_rows) < 5:
        cisc_rows.append(
            {
                "method": "cisc",
                "trial_id": None,
                "pred_answer": None,
                "confidence": None,
                "candidate_is_correct": None,
                "candidate_parse_failed": None,
                "raw_output": None,
            }
        )
    rows.extend(cisc_rows)
    return rows


def classify(sc_correct, cisc_correct):
    if (not sc_correct) and cisc_correct:
        return "SC错CISC对"
    if sc_correct and (not cisc_correct):
        return "前对后错"
    if (not sc_correct) and (not cisc_correct):
        return "两者都错"
    return None


def export_cases(
    sc_question_files,
    cisc_question_files,
    sc_candidate_files,
    cisc_candidate_files,
    out_dir,
):
    sc_q = build_question_map(sc_question_files)
    cisc_q = build_question_map(cisc_question_files)
    sc_cands = build_candidate_map(sc_candidate_files)
    cisc_cands = build_candidate_map(cisc_candidate_files)

    keys = sorted(set(sc_q.keys()) & set(cisc_q.keys()))
    rows = []

    for key in keys:
        sc = sc_q[key]
        cisc = cisc_q[key]
        sc_correct = bool(sc.get("is_correct"))
        cisc_correct = bool(cisc.get("is_correct"))
        label = classify(sc_correct, cisc_correct)
        if label is None:
            continue

        dataset, qid = key
        ten_raw = build_10_raw_answers(sc_cands.get(key, []), cisc_cands.get(key, []))

        row = {
            "分类": label,
            "问题来源": infer_problem_source(dataset),
            "类型": infer_problem_type(dataset, qid),
            "数据集": dataset,
            "id": qid,
            "题面": sc.get("question") or cisc.get("question"),
            "标准答案": sc.get("gold_answer") or cisc.get("gold_answer"),
            "sc_final_answer": sc.get("final_answer"),
            "sc_is_correct": sc_correct,
            "sc_vote_details": sc.get("vote_details"),
            "sc_all_parse_failed": sc.get("all_parse_failed"),
            "sc_raw_answer_selected": None,
            "sc_confidence_selected": None,
            "cisc_final_answer": cisc.get("final_answer"),
            "cisc_is_correct": cisc_correct,
            "cisc_vote_details": cisc.get("vote_details"),
            "cisc_all_parse_failed": cisc.get("all_parse_failed"),
            "cisc_raw_answer_selected": None,
            "cisc_confidence_selected": None,
            "raw_answers_10": ten_raw,
        }

        # 选中回答：按final_answer优先匹配；否则用trial_id最小
        sc_selected = None
        for cand in sc_cands.get(key, []):
            if cand.get("pred_answer") == sc.get("final_answer"):
                sc_selected = cand
                break
        if sc_selected is None and sc_cands.get(key):
            sc_selected = sc_cands[key][0]

        cisc_selected = None
        for cand in cisc_cands.get(key, []):
            if cand.get("pred_answer") == cisc.get("final_answer"):
                cisc_selected = cand
                break
        if cisc_selected is None and cisc_cands.get(key):
            cisc_selected = cisc_cands[key][0]

        if sc_selected is not None:
            row["sc_raw_answer_selected"] = sc_selected.get("raw_output")
            row["sc_confidence_selected"] = sc_selected.get("confidence")
        if cisc_selected is not None:
            row["cisc_raw_answer_selected"] = cisc_selected.get("raw_output")
            row["cisc_confidence_selected"] = cisc_selected.get("confidence")

        # 展平10条raw answer字段（csv友好）
        for i, item in enumerate(ten_raw, start=1):
            k = f"{i:02d}"
            row[f"raw_method_{k}"] = item.get("method")
            row[f"raw_trial_id_{k}"] = item.get("trial_id")
            row[f"raw_pred_answer_{k}"] = item.get("pred_answer")
            row[f"raw_confidence_{k}"] = item.get("confidence")
            row[f"raw_is_correct_{k}"] = item.get("candidate_is_correct")
            row[f"raw_parse_failed_{k}"] = item.get("candidate_parse_failed")
            row[f"raw_answer_{k}"] = item.get("raw_output")

        rows.append(row)

    rows.sort(key=lambda r: (r["分类"], r["数据集"], r["id"]))

    # 输出完整jsonl
    write_jsonl(out_dir / "sc_cisc_case_compare.jsonl", rows)

    # 分类别输出jsonl
    groups = defaultdict(list)
    for r in rows:
        groups[r["分类"]].append(r)
    for label, name in [
        ("SC错CISC对", "sc_wrong_cisc_right"),
        ("前对后错", "sc_right_cisc_wrong"),
        ("两者都错", "sc_wrong_cisc_wrong"),
    ]:
        write_jsonl(out_dir / f"{name}.jsonl", groups.get(label, []))

    # counts
    count_rows = []
    for label in ["SC错CISC对", "前对后错", "两者都错"]:
        subset = groups.get(label, [])
        count_rows.append({"分类": label, "数据集": "all", "count": len(subset)})
        by_ds = defaultdict(int)
        for r in subset:
            by_ds[r["数据集"]] += 1
        for ds, c in sorted(by_ds.items()):
            count_rows.append({"分类": label, "数据集": ds, "count": c})
    write_csv(out_dir / "sc_cisc_case_compare_counts.csv", count_rows, ["分类", "数据集", "count"])

    # csv（长文本截断，便于查看）
    csv_rows = []
    for r in rows:
        row = dict(r)
        row["题面"] = truncate_text(row["题面"], limit=800)
        row["sc_raw_answer_selected"] = truncate_text(row["sc_raw_answer_selected"], limit=1200)
        row["cisc_raw_answer_selected"] = truncate_text(row["cisc_raw_answer_selected"], limit=1200)
        row["sc_vote_details"] = json.dumps(row["sc_vote_details"], ensure_ascii=False)
        row["cisc_vote_details"] = json.dumps(row["cisc_vote_details"], ensure_ascii=False)
        row["raw_answers_10"] = json.dumps(row["raw_answers_10"], ensure_ascii=False)
        for i in range(1, 11):
            k = f"{i:02d}"
            row[f"raw_answer_{k}"] = truncate_text(row.get(f"raw_answer_{k}"), limit=1000)
        csv_rows.append(row)

    fieldnames = [
        "分类",
        "问题来源",
        "类型",
        "数据集",
        "id",
        "题面",
        "标准答案",
        "sc_final_answer",
        "sc_is_correct",
        "sc_vote_details",
        "sc_all_parse_failed",
        "sc_raw_answer_selected",
        "sc_confidence_selected",
        "cisc_final_answer",
        "cisc_is_correct",
        "cisc_vote_details",
        "cisc_all_parse_failed",
        "cisc_raw_answer_selected",
        "cisc_confidence_selected",
        "raw_answers_10",
    ]
    for i in range(1, 11):
        k = f"{i:02d}"
        fieldnames.extend(
            [
                f"raw_method_{k}",
                f"raw_trial_id_{k}",
                f"raw_pred_answer_{k}",
                f"raw_confidence_{k}",
                f"raw_is_correct_{k}",
                f"raw_parse_failed_{k}",
                f"raw_answer_{k}",
            ]
        )
    write_csv(out_dir / "sc_cisc_case_compare.csv", csv_rows, fieldnames)

    # README
    lines = [
        "# SC vs CISC Case Compare",
        "",
        "分类规则：",
        "- SC错CISC对",
        "- 前对后错（SC对CISC错）",
        "- 两者都错",
        "",
        "每条记录包含10条raw answer：",
        "- raw_answer_01~05：SC候选",
        "- raw_answer_06~10：CISC候选",
        "",
        "核心输出文件：",
        "- sc_cisc_case_compare.jsonl（完整，不截断）",
        "- sc_cisc_case_compare.csv（便于表格查看，长文本截断）",
        "- sc_cisc_case_compare_counts.csv（计数）",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/analysis/sc_cisc_case_compare")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = export_cases(
        sc_question_files=DEFAULT_SC_QUESTION_FILES,
        cisc_question_files=DEFAULT_CISC_QUESTION_FILES,
        sc_candidate_files=DEFAULT_SC_CANDIDATE_FILES,
        cisc_candidate_files=DEFAULT_CISC_CANDIDATE_FILES,
        out_dir=out_dir,
    )
    print(f"Exported {len(rows)} cases to {out_dir}")


if __name__ == "__main__":
    main()
