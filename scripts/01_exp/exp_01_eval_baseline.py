import argparse
import csv
import json
import math
import re
from pathlib import Path


def read_jsonl(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


FINAL_RE_LIST = [
    re.compile(
        r"(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?\s*(?:\d+\.\s*)?Final Answer\s*(?:\*\*)?\s*:\s*(.+)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?\s*(?:\d+\.\s*)?Final\s*(?:answer)?\s*(?:\*\*)?\s*[:：]\s*(.+)",
        flags=re.IGNORECASE,
    ),
]

BOXED_RE = re.compile(r"\\boxed\{(.+?)\}")


def extract_final_answer(text: str):
    if not text:
        return None

    # 先匹配 Final Answer 之后的内容
    for final_re in FINAL_RE_LIST:
        matches = final_re.findall(text)
        if matches:
            ans = matches[-1].strip()

            # 如果 Final Answer 后面一行是空的，后面才出现 boxed，则从后文取 boxed
            boxed = BOXED_RE.findall(ans)
            if boxed:
                ans = boxed[-1].strip()
            else:
                # 有些输出是：
                # ### Final Answer:
                # $$\boxed{6}$$
                tail_start = text.lower().rfind("final answer")
                tail = text[tail_start:] if tail_start != -1 else ans
                boxed_tail = BOXED_RE.findall(tail)
                if boxed_tail:
                    ans = boxed_tail[-1].strip()

            ans = re.split(r"\n\s*\n|Confidence:", ans, maxsplit=1)[0]
            ans = ans.replace("$$", "").replace("$", "")
            ans = ans.replace("**", "")
            ans = ans.strip()
            ans = ans.strip(" .。")
            return ans if ans else None

    # 如果没有 Final Answer，但全文最后出现 boxed，也尝试抽取 boxed
    boxed = BOXED_RE.findall(text)
    if boxed:
        ans = boxed[-1].strip()
        ans = ans.replace("$$", "").replace("$", "")
        ans = ans.strip(" .。")
        return ans if ans else None

    return None


def strip_boxed(text: str):
    text = str(text)

    # 处理 \boxed{...}
    match = re.search(r"\\boxed\{(.+?)\}", text)
    if match:
        return match.group(1)

    return text


def normalize_basic(text):
    if text is None:
        return None

    text = str(text)
    text = strip_boxed(text)

    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("\\,", "")
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("−", "-")
    text = text.replace("’", "'")
    text = text.replace("\\dfrac", "\\frac")
    text = text.replace("\\tfrac", "\\frac")

    text = text.strip()
    text = text.strip(" .。")

    # 去掉常见自然语言单位，主要服务 GSM8K
    text = re.sub(
        r"\b(dollars?|hours?|minutes?|bags?|students?|apples?|floors?|pieces?|seats?|lollipops?)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_number(text):
    if text is None:
        return None

    text = normalize_basic(text)

    # LaTeX 分数：\frac{a}{b}
    match = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", text)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator != 0:
            return numerator / denominator

    # 普通分数：a/b
    match = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", text)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator != 0:
            return numerator / denominator

    # 纯数字
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return float(text)

    # 如果混入短文本，取最后一个数字
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return float(nums[-1])

    return None


def is_correct(pred_answer, gold_answer, dataset_name):
    pred_norm = normalize_basic(pred_answer)
    gold_norm = normalize_basic(gold_answer)

    if pred_norm is None:
        return False, False

    if pred_norm == gold_norm:
        return True, False

    pred_num = parse_number(pred_norm)
    gold_num = parse_number(gold_norm)

    if pred_num is not None and gold_num is not None:
        return math.isclose(
            pred_num,
            gold_num,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ), False

    # MATH-500 中存在复杂 LaTeX 表达式，自动判断不了的样本标记为需复核
    needs_review = "math500" in str(dataset_name).lower()
    return False, needs_review


def summarize(rows):
    n = len(rows)
    correct = sum(1 for row in rows if row["is_correct"])
    parse_failed = sum(1 for row in rows if row["pred_answer"] is None)
    hit_max = sum(1 for row in rows if row.get("hit_max_new_tokens"))
    needs_review = sum(1 for row in rows if row.get("needs_review"))

    avg_output_tokens = sum(row.get("output_tokens", 0) for row in rows) / n if n else 0
    avg_latency_seconds = sum(row.get("latency_seconds", 0) for row in rows) / n if n else 0

    return {
        "dataset": rows[0]["dataset"] if rows else "",
        "method": rows[0]["method"] if rows else "",
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0,
        "parse_failed": parse_failed,
        "parse_failed_rate": parse_failed / n if n else 0,
        "needs_review": needs_review,
        "needs_review_rate": needs_review / n if n else 0,
        "hit_max": hit_max,
        "hit_max_rate": hit_max / n if n else 0,
        "avg_output_tokens": avg_output_tokens,
        "avg_latency_seconds": avg_latency_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "outputs/raw/baseline/gsm8k_platinum_direct.jsonl",
            "outputs/raw/baseline/gsm8k_platinum_cot.jsonl",
            "outputs/raw/baseline/math500_direct.jsonl",
            "outputs/raw/baseline/math500_cot.jsonl",
        ],
    )
    parser.add_argument("--parsed_dir", default="outputs/parsed/baseline")
    parser.add_argument("--summary_csv", default="outputs/tables/baseline_summary.csv")
    args = parser.parse_args()

    parsed_dir = Path(args.parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = []

    for input_path in args.inputs:
        input_file = Path(input_path)
        if not input_file.exists():
            print(f"skip missing file: {input_file}")
            continue

        raw_rows = read_jsonl(str(input_file))
        parsed_rows = []

        for row in raw_rows:
            pred_answer = extract_final_answer(row.get("raw_output", ""))
            correct, needs_review = is_correct(
                pred_answer=pred_answer,
                gold_answer=row.get("gold_answer"),
                dataset_name=row.get("dataset"),
            )

            parsed = {
                **row,
                "pred_answer": pred_answer,
                "is_correct": bool(correct),
                "needs_review": bool(needs_review),
            }
            parsed_rows.append(parsed)

        parsed_path = parsed_dir / f"{input_file.stem}_parsed.jsonl"
        with parsed_path.open("w", encoding="utf-8") as f:
            for row in parsed_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = summarize(parsed_rows)
        summary["source_file"] = str(input_file)
        summary["parsed_file"] = str(parsed_path)
        summaries.append(summary)

    fieldnames = [
        "dataset",
        "method",
        "n",
        "correct",
        "accuracy",
        "parse_failed",
        "parse_failed_rate",
        "needs_review",
        "needs_review_rate",
        "hit_max",
        "hit_max_rate",
        "avg_output_tokens",
        "avg_latency_seconds",
        "source_file",
        "parsed_file",
    ]

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)

    print("\nBaseline summary:")
    for summary in summaries:
        print(
            f"{summary['dataset']:18s} "
            f"{summary['method']:8s} "
            f"n={summary['n']:4d} "
            f"acc={summary['accuracy']:.4f} "
            f"parse_fail={summary['parse_failed_rate']:.4f} "
            f"review={summary['needs_review_rate']:.4f} "
            f"hit_max={summary['hit_max_rate']:.4f} "
            f"avg_tok={summary['avg_output_tokens']:.1f} "
            f"avg_time={summary['avg_latency_seconds']:.2f}s"
        )

    print(f"\nWrote summary to: {summary_path}")


if __name__ == "__main__":
    main()