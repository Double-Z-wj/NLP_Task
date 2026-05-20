import json
import random
import re
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()

def normalize_gsm8k_answer(answer_text: str) -> str:
    answer_text = str(answer_text)
    if "####" in answer_text:
        ans = answer_text.split("####")[-1]
    else:
        ans = answer_text
    ans = ans.strip()
    ans = ans.replace(",", "")
    ans = ans.replace("$", "")
    return normalize_space(ans)

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    ds = load_dataset("madrylab/gsm8k-platinum", "main", split="test")

    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "dataset": "gsm8k_platinum",
            "id": f"gsm8k_platinum_test_{i}",
            "question": ex["question"],
            "gold_answer": normalize_gsm8k_answer(ex["answer"]),
            "raw_answer": ex["answer"],
            "cleaning_status": ex.get("cleaning_status", None),
            "subject": None,
            "level": None,
            "source": "madrylab/gsm8k-platinum main test",
        })

    rng = random.Random(RANDOM_SEED)
    sampled_300 = rng.sample(rows, min(300, len(rows)))
    sampled_100 = sampled_300[:100]
    sampled_20 = sampled_300[:20]

    write_jsonl(OUT_DIR / "gsm8k_platinum_test_all.jsonl", rows)
    write_jsonl(OUT_DIR / "gsm8k_platinum_test_300_seed42.jsonl", sampled_300)
    write_jsonl(OUT_DIR / "gsm8k_platinum_test_100_seed42.jsonl", sampled_100)
    write_jsonl(OUT_DIR / "gsm8k_platinum_test_20_seed42.jsonl", sampled_20)

    print(f"GSM8K-Platinum all: {len(rows)}")
    print(f"GSM8K-Platinum 300 seed42: {len(sampled_300)}")
    print(f"GSM8K-Platinum 100 seed42: {len(sampled_100)}")
    print(f"GSM8K-Platinum 20 seed42: {len(sampled_20)}")

    status_counts = {}
    for row in rows:
        status = row.get("cleaning_status")
        status_counts[status] = status_counts.get(status, 0) + 1
    print("cleaning_status counts:", status_counts)

if __name__ == "__main__":
    main()
