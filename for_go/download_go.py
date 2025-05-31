import os
import re
import json
import tempfile
import subprocess
from tqdm import tqdm
from API_KEY import HF_KEY
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer

HUGGINGFACE_TOKEN = HF_KEY
MAX_RECORDS = 100_000
OUTPUT_JSONL = "go_code_filtered_streaming.jsonl"
MAX_TOKENS = 1024
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

def is_valid_go_function(code: str) -> bool:
    if re.search(r"type\s+\w+\s+interface\s*{", code):
        return False
    if re.search(r"type\s+\w+\s+struct\s*{", code):
        return False
    if re.match(r"^\s*package\s+\w+\s*$", code.strip()):
        return False
    if not re.search(r"func\s+\w+\s*\(", code):
        return False
    if re.search(r'import\s*\(([^)]+)\)', code):
        imports = re.findall(r'"([^"]+)"', code)
        for imp in imports:
            if not imp.startswith(("fmt", "math", "os", "strings", "bytes", "time")):
                return False
    elif re.search(r'import\s+"([^"]+)"', code):
        imp = re.search(r'import\s+"([^"]+)"', code).group(1)
        if not imp.startswith(("fmt", "math", "os", "strings", "bytes", "time")):
            return False
    return True

def can_compile(code: str) -> bool:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            go_file = os.path.join(tmpdir, "main.go")
            with open(go_file, "w") as f:
                f.write(code)
            result = subprocess.run(
                ["go", "tool", "compile", "-S", "main.go"],
                cwd=tmpdir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return result.returncode == 0
    except Exception:
        return False

def fits_model_context(code: str) -> bool:
    tokens = tokenizer(code, return_tensors="pt", truncation=False)
    return tokens.input_ids.shape[1] <= MAX_TOKENS

def main():
    login(HUGGINGFACE_TOKEN)

    print("Загружаем Go-раздел в потоковом режиме...")
    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/go",
        split="train",
        streaming=True
    )

    print("Фильтруем компилируемый Go-код...")
    count = 0

    with open(OUTPUT_JSONL, "w") as fout:
        for item in tqdm(ds, desc="Проверка", unit="файл"):
            if count >= MAX_RECORDS:
                break
            if item.get("ext") != "go" or item.get("lang", "").lower() != "go":
                continue

            code = item.get("content", "")
            if (
                is_valid_go_function(code)
                and fits_model_context(code)
                and can_compile(code)
            ):
                json.dump({"content": code}, fout)
                fout.write("\n")
                count += 1

    print(f"\nСохранено {count} Go-функций в {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
