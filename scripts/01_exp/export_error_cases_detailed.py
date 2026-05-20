import csv
import json
from collections import defaultdict
from pathlib import Path


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

SC_CISC_CANDIDATE_FILES = [
    Path("outputs/parsed/sc_cisc/gsm8k_platinum_sc_k5_candidates_parsed.jsonl"),
    Path("outputs/parsed/sc_cisc/gsm8k_platinum_cisc_k5_candidates_parsed.jsonl"),
    Path("outputs/parsed/sc_cisc/math500_sc_k5_candidates_parsed.jsonl"),
    Path("outputs/parsed/sc_cisc/math500_cisc_k5_candidates_parsed.jsonl"),
]

OUT_DIR = Path("outputs/analysis/error_cases_detailed")

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


def truncate_text(text, limit=300):
    if text is None:
        return None
    t = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if len(t) <= limit:
        return t
    return t[:limit] + "...(truncated)"


def load_method_question_records():
    by_key = defaultdict(dict)  # key=(dataset,id) -> method -> question-level row

    for path in BASELINE_PARSED_FILES:
        for row in read_jsonl(path):
            key = (row.get("dataset"), row.get("id"))
            by_key[key][row.get("method")] = row

    for path in SC_CISC_QUESTION_FILES:
        for row in read_jsonl(path):
            key = (row.get("dataset"), row.get("id"))
            by_key[key][row.get("method")] = row

    return by_key


def load_method_candidate_records():
    by_key_method = defaultdict(lambda: defaultdict(list))  # (dataset,id) -> method -> [rows]
    for path in SC_CISC_CANDIDATE_FILES:
        for row in read_jsonl(path):
            key = (row.get("dataset"), row.get("id"))
            method = row.get("method")
            by_key_method[key][method].append(row)

    for key in by_key_method:
        for method in by_key_method[key]:
            by_key_method[key][method].sort(key=lambda r: (r.get("trial_id", -1)))

    return by_key_method


def pick_selected_candidate(candidates, final_answer):
    if not candidates:
        return None
    if final_answer is None:
        return candidates[0]
    for row in candidates:
        if row.get("pred_answer") == final_answer:
            return row
    return candidates[0]


def serialize_candidates(candidates):
    simple = []
    for row in candidates:
        simple.append(
            {
                "trial_id": row.get("trial_id"),
                "pred_answer": row.get("pred_answer"),
                "confidence": row.get("confidence"),
                "candidate_is_correct": row.get("candidate_is_correct"),
                "candidate_parse_failed": row.get("candidate_parse_failed"),
                "raw_output": row.get("raw_output"),
            }
        )
    return simple


def build_detailed_case_rows():
    by_key_method = load_method_question_records()
    by_key_method_candidates = load_method_candidate_records()
    all_rows = []
    counts = []

    for case_slug, case_name, left_method, left_correct, right_method, right_correct in ERROR_CASE_DEFS:
        case_rows = []
        dataset_count = defaultdict(int)

        for key, methods in by_key_method.items():
            left = methods.get(left_method)
            right = methods.get(right_method)
            if left is None or right is None:
                continue
            if bool(left.get("is_correct")) != bool(left_correct):
                continue
            if bool(right.get("is_correct")) != bool(right_correct):
                continue

            dataset, qid = key
            direct = methods.get("direct")
            cot = methods.get("cot")
            sc = methods.get("sc")
            cisc = methods.get("cisc")

            sc_candidates = by_key_method_candidates[key].get("sc", [])
            cisc_candidates = by_key_method_candidates[key].get("cisc", [])
            sc_selected = pick_selected_candidate(sc_candidates, sc.get("final_answer") if sc else None)
            cisc_selected = pick_selected_candidate(cisc_candidates, cisc.get("final_answer") if cisc else None)

            question = None
            gold_answer = None
            for source in [direct, cot, sc, cisc]:
                if source and question is None:
                    question = source.get("question")
                if source and gold_answer is None:
                    gold_answer = source.get("gold_answer")

            row = {
                "case_type": case_name,
                "case_slug": case_slug,
                "dataset": dataset,
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "direct_pred": direct.get("pred_answer") if direct else None,
                "direct_is_correct": direct.get("is_correct") if direct else None,
                "direct_raw_output": direct.get("raw_output") if direct else None,
                "cot_pred": cot.get("pred_answer") if cot else None,
                "cot_is_correct": cot.get("is_correct") if cot else None,
                "cot_raw_output": cot.get("raw_output") if cot else None,
                "sc_pred": sc.get("final_answer") if sc else None,
                "sc_is_correct": sc.get("is_correct") if sc else None,
                "sc_vote_details": sc.get("vote_details") if sc else None,
                "sc_selected_candidate_raw_output": sc_selected.get("raw_output") if sc_selected else None,
                "sc_selected_candidate_trial_id": sc_selected.get("trial_id") if sc_selected else None,
                "sc_selected_candidate_confidence": sc_selected.get("confidence") if sc_selected else None,
                "sc_candidates": serialize_candidates(sc_candidates),
                "cisc_pred": cisc.get("final_answer") if cisc else None,
                "cisc_is_correct": cisc.get("is_correct") if cisc else None,
                "cisc_vote_details": cisc.get("vote_details") if cisc else None,
                "cisc_selected_candidate_raw_output": cisc_selected.get("raw_output") if cisc_selected else None,
                "cisc_selected_candidate_trial_id": cisc_selected.get("trial_id") if cisc_selected else None,
                "cisc_selected_candidate_confidence": cisc_selected.get("confidence") if cisc_selected else None,
                "cisc_candidates": serialize_candidates(cisc_candidates),
            }
            case_rows.append(row)
            dataset_count[dataset] += 1

        case_rows.sort(key=lambda r: (r["dataset"], r["id"]))
        all_rows.extend(case_rows)

        counts.append({"case_type": case_name, "dataset": "all", "count": len(case_rows)})
        for d, c in sorted(dataset_count.items()):
            counts.append({"case_type": case_name, "dataset": d, "count": c})

        # Write detailed jsonl per case type (full raw outputs)
        write_jsonl(OUT_DIR / f"{case_slug}.jsonl", case_rows)

        # Write readable csv per case type (raw outputs truncated for readability)
        csv_rows = []
        for row in case_rows:
            csv_rows.append(
                {
                    "case_type": row["case_type"],
                    "dataset": row["dataset"],
                    "id": row["id"],
                    "gold_answer": row["gold_answer"],
                    "direct_pred": row["direct_pred"],
                    "direct_is_correct": row["direct_is_correct"],
                    "cot_pred": row["cot_pred"],
                    "cot_is_correct": row["cot_is_correct"],
                    "sc_pred": row["sc_pred"],
                    "sc_is_correct": row["sc_is_correct"],
                    "cisc_pred": row["cisc_pred"],
                    "cisc_is_correct": row["cisc_is_correct"],
                    "question": truncate_text(row["question"], limit=500),
                    "direct_raw_output": truncate_text(row["direct_raw_output"], limit=500),
                    "cot_raw_output": truncate_text(row["cot_raw_output"], limit=500),
                    "sc_selected_candidate_raw_output": truncate_text(
                        row["sc_selected_candidate_raw_output"], limit=500
                    ),
                    "cisc_selected_candidate_raw_output": truncate_text(
                        row["cisc_selected_candidate_raw_output"], limit=500
                    ),
                    "sc_vote_details": json.dumps(row["sc_vote_details"], ensure_ascii=False),
                    "cisc_vote_details": json.dumps(row["cisc_vote_details"], ensure_ascii=False),
                }
            )

        write_csv(
            OUT_DIR / f"{case_slug}.csv",
            csv_rows,
            [
                "case_type",
                "dataset",
                "id",
                "gold_answer",
                "direct_pred",
                "direct_is_correct",
                "cot_pred",
                "cot_is_correct",
                "sc_pred",
                "sc_is_correct",
                "cisc_pred",
                "cisc_is_correct",
                "question",
                "direct_raw_output",
                "cot_raw_output",
                "sc_selected_candidate_raw_output",
                "cisc_selected_candidate_raw_output",
                "sc_vote_details",
                "cisc_vote_details",
            ],
        )

    write_jsonl(OUT_DIR / "all_error_cases_detailed.jsonl", all_rows)
    write_csv(
        OUT_DIR / "error_case_counts.csv",
        counts,
        ["case_type", "dataset", "count"],
    )
    return counts


def write_readme(counts):
    lines = []
    lines.append("# Error Cases Detailed Export")
    lines.append("")
    lines.append("每条错误案例包含：")
    lines.append("- direct/cot/sc/cisc 的最终答案与正确性")
    lines.append("- direct/cot 的完整 raw_output")
    lines.append("- sc/cisc 选中候选的 raw_output")
    lines.append("- sc/cisc 全部候选回答（jsonl 中的 sc_candidates/cisc_candidates）")
    lines.append("")
    lines.append("## Case Counts")
    for row in counts:
        if row["dataset"] == "all":
            lines.append(f"- {row['case_type']}: {row['count']}")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = build_detailed_case_rows()
    write_readme(counts)
    print("Detailed error cases exported to:", OUT_DIR)


if __name__ == "__main__":
    main()
