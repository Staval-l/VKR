import pandas as pd
import json

df = pd.read_parquet("train-00000-of-00001.parquet")
print("Columns:", df.columns)

if 'instruction' not in df.columns or 'output' not in df.columns:
    raise ValueError("Parquet file must contain 'instruction' and 'output' columns!")

with open("decompile-ghidra-100k.json", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        json_line = json.dumps({
            "instruction": row["instruction"],
            "output": row["output"]
        }, ensure_ascii=False)
        f.write(json_line + "\n")

print("Done! Saved as decompile-ghidra-100k.jsonl")
