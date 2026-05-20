import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def shorten(text, max_len=1200):
    if text is None:
        return ""
    text = str(text).replace("\r", "")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n...[TRUNCATED]..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/parsed/baseline/math500_cot_parsed.jsonl",
        help="Parsed jsonl file to inspect.",
    )
    parser.add_argument(
        "--out_jsonl",
        default="outputs/analysis/math500_cot_suspicious.jsonl",
        help="Output jsonl file for suspicious samples.",
    )
    parser.add_argument(
        "--out_csv",
        default="outputs/analysis/math500_cot_suspicious.csv",
        help="Output csv file for quick spreadsheet viewing.",
    )
    parser.add_argument(
        "--only_parse_fail",
        action="store_true",
        help="Export only parse failures. By default exports parse failures, hit_max, and needs_review.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)

    if args.only_parse_fail:
        suspicious = [r for r in rows if r.get("pred_answer") is None]
    else:
        suspicious = [
            r for r in rows
            if r.get("pred_answer") is None
            or r.get("hit_max_new_tokens")
            or r.get("needs_review")
        ]

    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in suspicious:
            item = {
                "dataset": r.get("dataset"),
                "id": r.get("id"),
                "method": r.get("method"),
                "gold_answer": r.get("gold_answer"),
                "pred_answer": r.get("pred_answer"),
                "is_correct": r.get("is_correct"),
                "parse_failed": r.get("pred_answer") is None,
                "hit_max_new_tokens": r.get("hit_max_new_tokens"),
                "needs_review": r.get("needs_review"),
                "output_tokens": r.get("output_tokens"),
                "latency_seconds": r.get("latency_seconds"),
                "question": r.get("question"),
                "raw_output": r.get("raw_output"),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    fieldnames = [
        "dataset",
        "id",
        "method",
        "gold_answer",
        "pred_answer",
        "is_correct",
        "parse_failed",
        "hit_max_new_tokens",
        "needs_review",
        "output_tokens",
        "latency_seconds",
        "question",
        "raw_output_short",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in suspicious:
            writer.writerow({
                "dataset": r.get("dataset"),
                "id": r.get("id"),
                "method": r.get("method"),
                "gold_answer": r.get("gold_answer"),
                "pred_answer": r.get("pred_answer"),
                "is_correct": r.get("is_correct"),
                "parse_failed": r.get("pred_answer") is None,
                "hit_max_new_tokens": r.get("hit_max_new_tokens"),
                "needs_review": r.get("needs_review"),
                "output_tokens": r.get("output_tokens"),
                "latency_seconds": r.get("latency_seconds"),
                "question": shorten(r.get("question"), 500),
                "raw_output_short": shorten(r.get("raw_output"), 1200),
            })

    print(f"input: {args.input}")
    print(f"total rows: {len(rows)}")
    print(f"exported suspicious rows: {len(suspicious)}")
    print(f"jsonl: {out_jsonl}")
    print(f"csv: {out_csv}")


if __name__ == "__main__":
    main()