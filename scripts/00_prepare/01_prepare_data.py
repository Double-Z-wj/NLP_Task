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
    ans = normalize_space(ans)
    return ans

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def prepare_gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="test")

    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "dataset": "gsm8k",
            "id": f"gsm8k_test_{i}",
            "question": ex["question"],
            "gold_answer": normalize_gsm8k_answer(ex["answer"]),
            "raw_answer": ex["answer"],
            "subject": None,
            "level": None,
            "source": "openai/gsm8k main test",
        })

    rng = random.Random(RANDOM_SEED)
    sampled_1000 = rng.sample(rows, min(1000, len(rows)))
    sampled_100 = sampled_1000[:100]

    write_jsonl(OUT_DIR / "gsm8k_test_all.jsonl", rows)
    write_jsonl(OUT_DIR / "gsm8k_test_1000_seed42.jsonl", sampled_1000)
    write_jsonl(OUT_DIR / "gsm8k_test_100_seed42.jsonl", sampled_100)

    print(f"GSM8K all: {len(rows)}")
    print(f"GSM8K 1000 seed42: {len(sampled_1000)}")
    print(f"GSM8K 100 seed42: {len(sampled_100)}")

def prepare_math500():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "dataset": "math500",
            "id": str(ex.get("unique_id", f"math500_test_{i}")),
            "question": ex["problem"],
            "gold_answer": normalize_space(ex["answer"]),
            "raw_solution": ex.get("solution", ""),
            "subject": ex.get("subject", None),
            "level": ex.get("level", None),
            "source": "HuggingFaceH4/MATH-500 test",
        })

    rng = random.Random(RANDOM_SEED)
    sampled_200 = rng.sample(rows, min(200, len(rows)))

    write_jsonl(OUT_DIR / "math500_test_all.jsonl", rows)
    write_jsonl(OUT_DIR / "math500_test_200_seed42.jsonl", sampled_200)

    print(f"MATH-500 all: {len(rows)}")
    print(f"MATH-500 200 seed42: {len(sampled_200)}")

def main():
    prepare_gsm8k()
    prepare_math500()

if __name__ == "__main__":
    main()
