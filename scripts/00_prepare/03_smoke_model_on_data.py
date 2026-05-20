import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./models/Qwen3-8B"

def read_jsonl(path, n=3):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows

def build_cisc_prompt(question):
    return (
        "Solve the math problem with concise reasoning.\n"
        "Use no more than 15 numbered steps.\n"
        "Do not repeat checks.\n"
        "Your final two lines must be exactly:\n"
        "Final Answer: <answer>\n"
        "Confidence: <integer from 0 to 100>\n\n"
        f"Problem:\n{question}"
    )

def generate_one(model, tokenizer, question):
    messages = [{"role": "user", "content": build_cisc_prompt(question)}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    rows = []
    rows += read_jsonl("data/gsm8k_platinum_test_20_seed42.jsonl", n=3)
    rows += read_jsonl("data/math500_test_200_seed42.jsonl", n=3)
    

    for row in rows:
        print("\n" + "=" * 80)
        print("dataset:", row["dataset"])
        print("id:", row["id"])
        print("gold:", row["gold_answer"])
        print("question:", row["question"][:300].replace("\n", " "))
        output = generate_one(model, tokenizer, row["question"])
        print("model output:\n", output)

if __name__ == "__main__":
    main()
