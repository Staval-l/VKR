import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer

INPUT_FILE = "go_llm4decompile_clean.jsonl"
MAX_TOKENS = 1024
SAMPLE_SIZE = 78786
MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-base"

def build_prompt(asm: str) -> str:
    return f"# This is the assembly code:\n{asm}\n# What is the source code?\n"

def fits_context(asm: str, code: str, tokenizer) -> bool:
    full_input = build_prompt(asm) + code
    tokens = tokenizer(full_input, return_tensors="pt", truncation=False)
    return tokens.input_ids.shape[1] <= MAX_TOKENS

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=True,
    )

    with open(INPUT_FILE, "r") as f:
        all_lines = f.readlines()

    print(f"Загружено {len(all_lines)} строк. Берём {SAMPLE_SIZE} случайных.")

    passed = 0
    failed = 0

    for line in tqdm(random.sample(all_lines, min(SAMPLE_SIZE, len(all_lines))), desc="Проверка токенов"):
        try:
            obj = json.loads(line)
            asm = obj.get("asm", "")
            code = obj.get("code", "")
            if fits_context(asm, code, tokenizer):
                passed += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    print(f"\nПрошли проверку: {passed}/{SAMPLE_SIZE}")
    print(f"Не прошли: {failed}/{SAMPLE_SIZE}")

if __name__ == "__main__":
    main()
