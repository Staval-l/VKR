from API_KEY import API_KEY
import os
import json
import tempfile
import subprocess
from openai import OpenAI
from tqdm import trange
import argparse
import re
import difflib

os.environ['http_proxy'] = 'socks5h://127.0.0.1:10808'
os.environ['https_proxy'] = 'socks5h://127.0.0.1:10808'

API = API_KEY
DATA_PATH = "decompile-eval-executable-gcc-obj.json"
MODEL = "gpt-4o"

client = OpenAI(api_key=API_KEY)
OPT = ["O0", "O1", "O2", "O3"]


def extract_code_block(text):
    match = re.search(r"```c?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def compute_edit_distance(a: str, b: str) -> int:
    return sum(1 for _ in difflib.ndiff(a.split(), b.split()) if _.startswith('+ ') or _.startswith('- '))

def evaluate_func(c_func, c_test, c_func_decompile):
    if not re.search(r"\bfunc0\s*\(", c_func_decompile):
        return 0, 0
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            c_file = os.path.join(tmpdir, "tmp.c")
            exe_file = os.path.join(tmpdir, "tmp.out")
            with open(c_file, "w") as f:
                f.write(c_func_decompile + "\n" + c_test)
            compile_result = subprocess.run(
                f"gcc {c_file} -o {exe_file} -lm", shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=10
            )
            if compile_result.returncode != 0:
                return 0, 0
            run_result = subprocess.run([exe_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            return 1, int(run_result.returncode == 0)
    except Exception:
        return 0, 0

parser = argparse.ArgumentParser()
parser.add_argument("--test_one", action="store_true")
args = parser.parse_args()

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data_all = json.load(f)

start_idx = 0
if os.path.exists("decompiled_outputs.txt"):
    with open("decompiled_outputs.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("=== Sample"):
                start_idx += 1

if args.test_one:
    print("Тестируем только один пример...\n")
    item = data_all[start_idx]
    c_func = item["c_func"]
    c_test = item["c_test"]
    asm = item["input_asm_prompt"]
    opt_state = item["type"]

    prompt = f"# This is the assembly code with {opt_state} optimization:\n{asm.strip()}\n# What is the source code?\n"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that converts assembly code into C."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    raw_output = response.choices[0].message.content.strip()
    c_func_decompile = extract_code_block(raw_output)

    print("\nGenerated:\n", c_func_decompile)
    dist = compute_edit_distance(c_func, c_func_decompile)
    print(f"\nEdit distance: {dist}")
    flag_compile, flag_run = evaluate_func(c_func, c_test, c_func_decompile)
    print(f"\nCompile OK: {flag_compile}, Run OK: {flag_run}")
    exit(0)

NUM = len(data_all) // 4
num_compile = {opt: 0 for opt in OPT}
num_run = {opt: 0 for opt in OPT}
edit_distances = {opt: [] for opt in OPT}

with open("decompiled_outputs.txt", "a", encoding="utf-8") as fout:
    for idx in trange(start_idx, len(data_all)):
        item = data_all[idx]
        c_func = item["c_func"]
        c_test = item["c_test"]
        asm = item["input_asm_prompt"]
        opt_state = item["type"]

        prompt = f"# This is the assembly code with {opt_state} optimization:\n{asm.strip()}\n# What is the source code?\n"
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that converts assembly code into C."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024,
                timeout=30
            )
            raw_output = response.choices[0].message.content.strip()
            c_func_decompile = extract_code_block(raw_output)
        except Exception as e:
            print(f"OpenAI error at idx={idx}: {e}")
            c_func_decompile = ""

        fout.write(f"=== Sample {idx} | opt: {opt_state} ===\n")
        fout.write(c_func_decompile + "\n\n")

        dist = compute_edit_distance(c_func, c_func_decompile)
        edit_distances[opt_state].append(dist)

        flag_compile, flag_run = evaluate_func(c_func, c_test, c_func_decompile)
        num_compile[opt_state] += flag_compile
        num_run[opt_state] += flag_run

total_compile = sum(num_compile.values())
total_run = sum(num_run.values())

with open("results.txt", "a", encoding="utf-8") as f:
    for opt in OPT:
        compile_rate = num_compile[opt] / NUM
        run_rate = num_run[opt] / NUM
        avg_dist = sum(edit_distances[opt]) / len(edit_distances[opt]) if edit_distances[opt] else -1
        f.write(f"model:{MODEL}, opt:{opt}, compile_rate:{compile_rate:.4f}, run_rate:{run_rate:.4f}, avg_edit_distance:{avg_dist:.2f}\n")
    f.write(f"Total compile OK: {total_compile}/{len(data_all)}\n")
    f.write(f"Total run OK: {total_run}/{len(data_all)}\n")

print("Завершено. Продолжение было с idx =", start_idx)
