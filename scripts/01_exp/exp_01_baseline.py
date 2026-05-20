import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "./models/Qwen3-8B"

DATASETS = {
    "gsm8k_platinum": {
        "path": "data/gsm8k_platinum_test_300_seed42.jsonl",
        "max_steps": 10,
        "direct_max_new_tokens": 128,
        "cot_max_new_tokens": 1024,
    },
    "math500": {
        "path": "data/math500_test_200_seed42.jsonl",
        "max_steps": 15,
        "direct_max_new_tokens": 128,
        "cot_max_new_tokens": 2048,
    },
}

METHODS = ["direct", "cot"]


def read_jsonl(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_done_ids(output_path: Path):
    done = set()
    if not output_path.exists():
        return done

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add(row["id"])
            except Exception:
                continue

    return done


def build_prompt(method: str, question: str, max_steps: int):
    if method == "direct":
        return (
            "Solve the following math problem.\n"
            "You must answer directly.\n"
            "Do not explain.\n"
            "Do not show your reasoning.\n"
            "Your entire response must contain exactly one line:\n"
            "Final Answer: <answer>\n\n"
            f"Problem:\n{question}"
        )

    if method == "cot":
        return (
            "Solve the following math problem with concise reasoning.\n"
            f"Use no more than {max_steps} numbered steps.\n"
            "Do not repeat checks.\n"
            "Your last line must be exactly:\n"
            "Final Answer: <answer>\n\n"
            f"Problem:\n{question}"
        )

    raise ValueError(f"Unknown method: {method}")


def get_max_new_tokens(method: str, dataset_cfg: dict):
    if method == "direct":
        return dataset_cfg["direct_max_new_tokens"]
    if method == "cot":
        return dataset_cfg["cot_max_new_tokens"]
    raise ValueError(f"Unknown method: {method}")


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int):
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_tokens = inputs["input_ids"].shape[-1]

    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    latency_seconds = time.time() - start

    new_tokens = output_ids[0][input_tokens:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "raw_output": output_text,
        "input_tokens": input_tokens,
        "output_tokens": len(new_tokens),
        "latency_seconds": latency_seconds,
        "hit_max_new_tokens": len(new_tokens) >= max_new_tokens,
    }


def run_one_setting(model, tokenizer, dataset_name: str, method: str, overwrite: bool):
    dataset_cfg = DATASETS[dataset_name]
    input_path = dataset_cfg["path"]
    rows = read_jsonl(input_path)

    output_dir = Path("outputs/raw/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}_{method}.jsonl"

    if overwrite and output_path.exists():
        output_path.unlink()

    done_ids = load_done_ids(output_path)

    max_new_tokens = get_max_new_tokens(method, dataset_cfg)

    print(f"\nRunning dataset={dataset_name}, method={method}")
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"num_rows={len(rows)}")
    print(f"max_new_tokens={max_new_tokens}")
    print(f"already_done={len(done_ids)}")

    with output_path.open("a", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            if row["id"] in done_ids:
                continue

            prompt = build_prompt(
                method=method,
                question=row["question"],
                max_steps=dataset_cfg["max_steps"],
            )

            result = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )

            record = {
                "dataset": row["dataset"],
                "id": row["id"],
                "method": method,
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "prompt": prompt,
                "max_steps": dataset_cfg["max_steps"],
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                **result,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            print(
                f"[{idx + 1}/{len(rows)}] "
                f"{dataset_name} {method} "
                f"id={row['id']} "
                f"out_tokens={record['output_tokens']} "
                f"hit_max={record['hit_max_new_tokens']} "
                f"time={record['latency_seconds']:.2f}s"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["all", "gsm8k_platinum", "math500"],
        default="all",
    )
    parser.add_argument(
        "--method",
        choices=["all", "direct", "cot"],
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    methods = METHODS if args.method == "all" else [args.method]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    for dataset_name in datasets:
        for method in methods:
            run_one_setting(
                model=model,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                method=method,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()