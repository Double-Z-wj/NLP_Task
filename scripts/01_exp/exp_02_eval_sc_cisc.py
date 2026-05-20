import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUTS = [
    "outputs/raw/sc_cisc/gsm8k_platinum_sc_k5.jsonl",
    "outputs/raw/sc_cisc/gsm8k_platinum_cisc_k5.jsonl",
    "outputs/raw/sc_cisc/math500_sc_k5.jsonl",
    "outputs/raw/sc_cisc/math500_cisc_k5.jsonl",
]


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
CONF_RE = re.compile(
    r"(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?\s*(?:\d+\.\s*)?Confidence\s*(?:\*\*)?\s*[:：]\s*(\d+)",
    flags=re.IGNORECASE,
)


def read_jsonl(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_final_answer(text: str):
    if not text:
        return None

    for final_re in FINAL_RE_LIST:
        matches = final_re.findall(text)
        if matches:
            ans = matches[-1].strip()

            boxed = BOXED_RE.findall(ans)
            if boxed:
                ans = boxed[-1].strip()
            else:
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

    boxed = BOXED_RE.findall(text)
    if boxed:
        ans = boxed[-1].strip()
        ans = ans.replace("$$", "").replace("$", "")
        ans = ans.strip(" .。")
        return ans if ans else None

    return None


def extract_confidence(text: str):
    if not text:
        return None

    matches = CONF_RE.findall(text)
    if not matches:
        return None

    try:
        value = int(matches[-1])
    except ValueError:
        return None

    return max(0, min(100, value))


def strip_boxed(text: str):
    text = str(text)
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
    text = text.replace("\\dfrac", "\\frac")
    text = text.replace("\\tfrac", "\\frac")
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("−", "-")
    text = text.replace("’", "'")
    text = text.strip()
    text = text.strip(" .。")

    text = re.sub(
        r"\b(dollars?|hours?|minutes?|bags?|students?|apples?|floors?|pieces?|seats?|lollipops?)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_vote(text):
    text = normalize_basic(text)
    if text is None:
        return None

    text = text.lower()
    text = text.replace(" ", "")

    frac = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", text)
    if frac:
        numerator = frac.group(1)
        denominator = frac.group(2)
        return f"{numerator}/{denominator}"

    return text


def parse_number(text):
    if text is None:
        return None

    text = normalize_basic(text)

    match = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", text)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator != 0:
            return numerator / denominator

    match = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", text)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator != 0:
            return numerator / denominator

    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return float(text)

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
        return math.isclose(pred_num, gold_num, rel_tol=1e-9, abs_tol=1e-9), False

    needs_review = "math500" in str(dataset_name).lower()
    return False, needs_review


def majority_vote(candidates):
    votes = Counter()
    first_seen = {}

    for idx, candidate in enumerate(candidates):
        vote_key = candidate.get("vote_key")
        if vote_key is None:
            continue

        votes[vote_key] += 1
        if vote_key not in first_seen:
            first_seen[vote_key] = idx

    if not votes:
        return None, None, {}

    max_votes = max(votes.values())
    tied = [key for key, value in votes.items() if value == max_votes]
    winner_key = min(tied, key=lambda key: first_seen[key])

    winner_answer = None
    for candidate in candidates:
        if candidate.get("vote_key") == winner_key:
            winner_answer = candidate.get("pred_answer")
            break

    return winner_key, winner_answer, dict(votes)


def confidence_weighted_vote(candidates, missing_confidence_value=50):
    scores = defaultdict(float)
    first_seen = {}

    for idx, candidate in enumerate(candidates):
        vote_key = candidate.get("vote_key")
        if vote_key is None:
            continue

        confidence = candidate.get("confidence")
        if confidence is None:
            confidence = missing_confidence_value

        scores[vote_key] += float(confidence)
        if vote_key not in first_seen:
            first_seen[vote_key] = idx

    if not scores:
        return None, None, {}

    max_score = max(scores.values())
    tied = [key for key, value in scores.items() if value == max_score]
    winner_key = min(tied, key=lambda key: first_seen[key])

    winner_answer = None
    for candidate in candidates:
        if candidate.get("vote_key") == winner_key:
            winner_answer = candidate.get("pred_answer")
            break

    return winner_key, winner_answer, dict(scores)


def parse_candidates(rows):
    parsed = []

    for row in rows:
        raw_output = row.get("raw_output", "")
        method = row.get("method")

        pred_answer = row.get("pred_answer")
        if pred_answer is None:
            pred_answer = extract_final_answer(raw_output)

        confidence = row.get("confidence")
        if method == "cisc" and confidence is None:
            confidence = extract_confidence(raw_output)

        vote_key = normalize_for_vote(pred_answer)

        candidate_correct, candidate_needs_review = is_correct(
            pred_answer=pred_answer,
            gold_answer=row.get("gold_answer"),
            dataset_name=row.get("dataset"),
        )

        parsed.append({
            **row,
            "pred_answer": pred_answer,
            "confidence": confidence,
            "vote_key": vote_key,
            "candidate_is_correct": bool(candidate_correct),
            "candidate_needs_review": bool(candidate_needs_review),
            "candidate_parse_failed": pred_answer is None,
        })

    return parsed


def group_by_question(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["id"], row["method"], int(row["k"]))
        groups[key].append(row)
    return groups


def aggregate_questions(candidate_rows, missing_confidence_value=50):
    groups = group_by_question(candidate_rows)
    question_rows = []

    for (dataset, qid, method, k), candidates in groups.items():
        candidates = sorted(candidates, key=lambda r: int(r["trial_id"]))

        if method == "sc":
            winner_key, final_answer, vote_details = majority_vote(candidates)
            aggregation = "majority_vote"
        elif method == "cisc":
            winner_key, final_answer, vote_details = confidence_weighted_vote(
                candidates,
                missing_confidence_value=missing_confidence_value,
            )
            aggregation = "confidence_weighted_vote"
        else:
            raise ValueError(f"Unknown method: {method}")

        gold_answer = candidates[0].get("gold_answer")
        question = candidates[0].get("question")

        final_correct, final_needs_review = is_correct(
            pred_answer=final_answer,
            gold_answer=gold_answer,
            dataset_name=dataset,
        )

        n_candidates = len(candidates)
        candidate_parse_failed = sum(1 for c in candidates if c.get("candidate_parse_failed"))
        candidate_hit_max = sum(1 for c in candidates if c.get("hit_max_new_tokens"))
        candidate_correct = sum(1 for c in candidates if c.get("candidate_is_correct"))
        missing_confidence = sum(
            1 for c in candidates
            if method == "cisc" and c.get("confidence") is None
        )

        total_output_tokens = sum(c.get("output_tokens", 0) for c in candidates)
        total_latency_seconds = sum(c.get("latency_seconds", 0) for c in candidates)

        question_rows.append({
            "dataset": dataset,
            "id": qid,
            "method": method,
            "k": k,
            "aggregation": aggregation,
            "question": question,
            "gold_answer": gold_answer,
            "final_answer": final_answer,
            "final_vote_key": winner_key,
            "is_correct": bool(final_correct),
            "needs_review": bool(final_needs_review),
            "all_parse_failed": final_answer is None,
            "n_candidates": n_candidates,
            "candidate_parse_failed": candidate_parse_failed,
            "candidate_parse_failed_rate": candidate_parse_failed / n_candidates if n_candidates else 0,
            "candidate_hit_max": candidate_hit_max,
            "candidate_hit_max_rate": candidate_hit_max / n_candidates if n_candidates else 0,
            "candidate_correct": candidate_correct,
            "candidate_correct_rate": candidate_correct / n_candidates if n_candidates else 0,
            "missing_confidence": missing_confidence,
            "missing_confidence_rate": missing_confidence / n_candidates if n_candidates else 0,
            "vote_details": vote_details,
            "total_output_tokens": total_output_tokens,
            "avg_output_tokens_per_candidate": total_output_tokens / n_candidates if n_candidates else 0,
            "total_latency_seconds": total_latency_seconds,
            "avg_latency_per_candidate": total_latency_seconds / n_candidates if n_candidates else 0,
        })

    return sorted(question_rows, key=lambda r: (r["dataset"], r["method"], r["id"]))


def summarize_questions(question_rows):
    groups = defaultdict(list)
    for row in question_rows:
        key = (row["dataset"], row["method"], row["k"])
        groups[key].append(row)

    summaries = []

    for (dataset, method, k), rows in sorted(groups.items()):
        n = len(rows)
        correct = sum(1 for r in rows if r["is_correct"])
        needs_review = sum(1 for r in rows if r["needs_review"])
        all_parse_failed = sum(1 for r in rows if r["all_parse_failed"])

        avg_total_tokens = sum(r["total_output_tokens"] for r in rows) / n if n else 0
        avg_total_latency = sum(r["total_latency_seconds"] for r in rows) / n if n else 0

        avg_candidate_tokens = sum(r["avg_output_tokens_per_candidate"] for r in rows) / n if n else 0
        avg_candidate_latency = sum(r["avg_latency_per_candidate"] for r in rows) / n if n else 0

        avg_candidate_parse_fail_rate = sum(r["candidate_parse_failed_rate"] for r in rows) / n if n else 0
        avg_candidate_hit_max_rate = sum(r["candidate_hit_max_rate"] for r in rows) / n if n else 0
        avg_missing_confidence_rate = sum(r["missing_confidence_rate"] for r in rows) / n if n else 0

        summaries.append({
            "dataset": dataset,
            "method": method,
            "k": k,
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else 0,
            "needs_review": needs_review,
            "needs_review_rate": needs_review / n if n else 0,
            "all_parse_failed": all_parse_failed,
            "all_parse_failed_rate": all_parse_failed / n if n else 0,
            "avg_candidate_parse_fail_rate": avg_candidate_parse_fail_rate,
            "avg_candidate_hit_max_rate": avg_candidate_hit_max_rate,
            "avg_missing_confidence_rate": avg_missing_confidence_rate,
            "avg_total_output_tokens_per_question": avg_total_tokens,
            "avg_total_latency_seconds_per_question": avg_total_latency,
            "avg_output_tokens_per_candidate": avg_candidate_tokens,
            "avg_latency_seconds_per_candidate": avg_candidate_latency,
        })

    return summaries


def write_summary_csv(path: Path, summaries):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "method",
        "k",
        "n",
        "correct",
        "accuracy",
        "needs_review",
        "needs_review_rate",
        "all_parse_failed",
        "all_parse_failed_rate",
        "avg_candidate_parse_fail_rate",
        "avg_candidate_hit_max_rate",
        "avg_missing_confidence_rate",
        "avg_total_output_tokens_per_question",
        "avg_total_latency_seconds_per_question",
        "avg_output_tokens_per_candidate",
        "avg_latency_seconds_per_candidate",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--parsed_dir", default="outputs/parsed/sc_cisc")
    parser.add_argument("--summary_csv", default="outputs/tables/sc_cisc_summary.csv")
    parser.add_argument("--missing_confidence_value", type=int, default=50)
    args = parser.parse_args()

    all_candidates = []
    all_questions = []

    parsed_dir = Path(args.parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    for input_path in args.inputs:
        path = Path(input_path)
        if not path.exists():
            print(f"skip missing file: {path}")
            continue

        raw_rows = read_jsonl(str(path))
        candidate_rows = parse_candidates(raw_rows)
        question_rows = aggregate_questions(
            candidate_rows,
            missing_confidence_value=args.missing_confidence_value,
        )

        candidate_out = parsed_dir / f"{path.stem}_candidates_parsed.jsonl"
        question_out = parsed_dir / f"{path.stem}_questions.jsonl"

        write_jsonl(candidate_out, candidate_rows)
        write_jsonl(question_out, question_rows)

        print(f"\nInput: {path}")
        print(f"candidates: {len(candidate_rows)} -> {candidate_out}")
        print(f"questions: {len(question_rows)} -> {question_out}")

        all_candidates.extend(candidate_rows)
        all_questions.extend(question_rows)

    summaries = summarize_questions(all_questions)
    summary_path = Path(args.summary_csv)
    write_summary_csv(summary_path, summaries)

    print("\nSC/CISC summary:")
    for s in summaries:
        print(
            f"{s['dataset']:18s} "
            f"{s['method']}@{s['k']} "
            f"n={s['n']:4d} "
            f"acc={s['accuracy']:.4f} "
            f"all_parse_fail={s['all_parse_failed_rate']:.4f} "
            f"cand_parse_fail={s['avg_candidate_parse_fail_rate']:.4f} "
            f"hit_max={s['avg_candidate_hit_max_rate']:.4f} "
            f"miss_conf={s['avg_missing_confidence_rate']:.4f} "
            f"tok/q={s['avg_total_output_tokens_per_question']:.1f} "
            f"time/q={s['avg_total_latency_seconds_per_question']:.2f}s"
        )

    print(f"\nWrote summary to: {summary_path}")


if __name__ == "__main__":
    main()