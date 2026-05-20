import json
from pathlib import Path
from collections import Counter

FILES = [
    "data/gsm8k_test_100_seed42.jsonl",
    "data/gsm8k_test_1000_seed42.jsonl",
    "data/gsm8k_test_all.jsonl",
    "data/math500_test_200_seed42.jsonl",
    "data/math500_test_all.jsonl",
]

def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def check(path):
    rows = read_jsonl(path)
    print("\n" + "=" * 80)
    print(path)
    print("num rows:", len(rows))

    first = rows[0]
    print("keys:", sorted(first.keys()))
    print("dataset:", first["dataset"])
    print("id:", first["id"])
    print("question:", first["question"][:300].replace("\n", " "))
    print("gold_answer:", first["gold_answer"])

    missing_question = sum(1 for r in rows if not r.get("question"))
    missing_answer = sum(1 for r in rows if not r.get("gold_answer"))
    print("missing_question:", missing_question)
    print("missing_gold_answer:", missing_answer)

    if first["dataset"] == "math500":
        print("subject counts:", Counter(r.get("subject") for r in rows))
        print("level counts:", Counter(r.get("level") for r in rows))

def main():
    for path in FILES:
        check(path)

if __name__ == "__main__":
    main()
