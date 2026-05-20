import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./models/Qwen3-8B"

def main():
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": (
                "Please solve the following math problem step by step. "
                "At the end, output exactly one line in this format: Final Answer: <answer>\n\n"
                "Problem: If Tom has 3 apples and buys 4 more, how many apples does he have?"
            ),
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))

if __name__ == "__main__":
    main()
