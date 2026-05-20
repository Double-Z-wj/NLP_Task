import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "./models/Qwen3-8B"

DATASETS = {
    "gsm8k_platinum": {
        "path": "data/gsm8k_platinum_test_300_seed42.jsonl",
        "max_steps": 10,
        "max_new_tokens": 1024,
    },
    "math500": {
        "path": "data/math500_test_200_seed42.jsonl",
        "max_steps": 15,
        "max_new_tokens": 2048,
    },
}

METHODS = ["sc", "cisc"]

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


def load_done_keys(output_path: Path):
    done = set()
    if not output_path.exists():
        return done

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add((row["id"], int(row["trial_id"])))
            except Exception:
                continue

    return done


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

    if value < 0:
        value = 0
    if value > 100:
        value = 100

    return value


def build_prompt(method: str, question: str, max_steps: int):
    if method == "sc":
        return (
            "Solve the following math problem with concise reasoning.\n"
            f"Use no more than {max_steps} numbered steps.\n"
            "Do not repeat checks.\n"
            "Your last line must be exactly:\n"
            "Final Answer: <answer>\n\n"
            f"Problem:\n{question}"
        )

    if method == "cisc":
        return (
            "Solve the following math problem with concise reasoning.\n"
            f"Use no more than {max_steps} numbered steps.\n"
            "Do not repeat checks.\n"
            "Your final two lines must be exactly:\n"
            "Final Answer: <answer>\n"
            "Confidence: <integer from 0 to 100>\n\n"
            "Confidence should reflect your uncertainty.\n"
            "Use 100 only if the computation is short and fully verified.\n"
            "Use 60-80 if the reasoning involves several algebraic or arithmetic steps.\n"
            "Use below 60 if any step is uncertain.\n\n"
            f"Problem:\n{question}"
        )

    raise ValueError(f"Unknown method: {method}")


def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=1,
        )

    latency_seconds = time.time() - start

    new_tokens = output_ids[0][input_tokens:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "raw_output": raw_output,
        "input_tokens": input_tokens,
        "output_tokens": len(new_tokens),
        "latency_seconds": latency_seconds,
        "hit_max_new_tokens": len(new_tokens) >= max_new_tokens,
    }


def run_one_setting(
    model,
    tokenizer,
    dataset_name: str,
    method: str,
    k: int,
    limit,
    overwrite: bool,
    base_seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
):
    dataset_cfg = DATASETS[dataset_name]
    rows = read_jsonl(dataset_cfg["path"])

    if limit is not None:
        rows = rows[:limit]

    output_dir = Path("outputs/raw/sc_cisc")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}_{method}_k{k}.jsonl"

    if overwrite and output_path.exists():
        output_path.unlink()

    done_keys = load_done_keys(output_path)

    print(f"\nRunning dataset={dataset_name}, method={method}, k={k}")
    print(f"input={dataset_cfg['path']}")
    print(f"output={output_path}")
    print(f"num_questions={len(rows)}")
    print(f"max_steps={dataset_cfg['max_steps']}")
    print(f"max_new_tokens={dataset_cfg['max_new_tokens']}")
    print(f"temperature={temperature}, top_p={top_p}, top_k={top_k}")
    print(f"already_done_candidates={len(done_keys)}")

    with output_path.open("a", encoding="utf-8") as f:
        for q_idx, row in enumerate(rows):
            prompt = build_prompt(
                method=method,
                question=row["question"],
                max_steps=dataset_cfg["max_steps"],
            )

            for trial_id in range(k):
                key = (row["id"], trial_id)
                if key in done_keys:
                    continue

                sample_seed = base_seed + q_idx * 1000 + trial_id

                result = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=dataset_cfg["max_new_tokens"],
                    seed=sample_seed,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )

                pred_answer = extract_final_answer(result["raw_output"])
                confidence = extract_confidence(result["raw_output"]) if method == "cisc" else None

                record = {
                    "dataset": row["dataset"],
                    "id": row["id"],
                    "method": method,
                    "k": k,
                    "trial_id": trial_id,
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "prompt": prompt,
                    "max_steps": dataset_cfg["max_steps"],
                    "max_new_tokens": dataset_cfg["max_new_tokens"],
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "seed": sample_seed,
                    "pred_answer": pred_answer,
                    "confidence": confidence,
                    **result,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                print(
                    f"[{q_idx + 1}/{len(rows)}] "
                    f"{dataset_name} {method}@{k} "
                    f"id={row['id']} "
                    f"trial={trial_id} "
                    f"pred={pred_answer} "
                    f"conf={confidence} "
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
        choices=["all", "sc", "cisc"],
        default="all",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)

    parser.add_argument("--model_path", default=MODEL_PATH)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    methods = METHODS if args.method == "all" else [args.method]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
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
                k=args.k,
                limit=args.limit,
                overwrite=args.overwrite,
                base_seed=args.seed,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )


if __name__ == "__main__":
    main()