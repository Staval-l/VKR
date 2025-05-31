from datasets import load_dataset, Dataset
import json
import os
from tqdm import tqdm

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'

dataset = load_dataset("Neo111x/decompile-dataset-large-asm", split="train")

dataset = dataset.filter(lambda x: x.get("Compiler") == "gcc")

examples_per_opt = 25_000
opt_levels = ["O0", "O1", "O2", "O3"]

balanced_samples = []

for opt_level in opt_levels:
    print(f"Обработка opt={opt_level}")
    subset = dataset.filter(lambda x: x.get("Optimization") == opt_level)
    if len(subset) < examples_per_opt:
        print(f"Недостаточно примеров для {opt_level}: найдено {len(subset)}, нужно {examples_per_opt}")
    subset = subset.shuffle(seed=42).select(range(min(examples_per_opt, len(subset))))
    balanced_samples.extend(subset)

final_dataset = Dataset.from_list(balanced_samples)

output_file = "decompile-asm-llm4decompile-100k-balanced.json"

with open(output_file, "w", encoding="utf-8") as f:
    for item in tqdm(final_dataset):
        json_obj = {
            "instruction": f"# This is the assembly code:\n{item['Decompiled Source']}\n# What is the source code?\n",
            "output": item["Original Source"]
        }
        f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print(f"Сохранено {len(final_dataset)} примеров в {output_file}")
