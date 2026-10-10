"""Datasets, prompt templates and the fixed dev/test split (brief §3).

Prompt templates are copied verbatim from the DFlash repo's original eval code
(z-lab/dflash commit bb48ef9, model/utils.py: `aime24` and `livecodebench`).
The current DFlash HEAD no longer ships AIME/LCB loaders.
"""
import json
import random

MATH_TEMPLATE = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

LCB_FILE = "test6.jsonl"  # LiveCodeBench v6 = problems added in release_v6 (2025-01..2025-04)
N_LCB = 32
N_DEV = {"aime": 6, "lcb": 6}
SEED = 0


def format_lcb(doc):
    system_prompt = (
        "You are an expert Python programmer. You will be given a question (problem specification) "
        "and will generate a correct Python program that matches the specification and passes all tests. "
        "You will NOT return anything except for the program"
    )
    question_block = f"### Question:\n{doc['question_content']}"
    if doc.get("starter_code"):
        format_message = "### Format: Use the following code structure:"
        code_block = f"```python\n{doc['starter_code']}\n```"
    else:
        format_message = "### Format: Write your code in the following format:"
        code_block = "```python\n# YOUR CODE HERE\n```"
    answer_footer = "### Answer: (use the provided format with backticks)"
    return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"


def load_problems():
    """Return list of dicts: trace_id, dataset, split, prompt (user message text)."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    aime = []
    for part in ("I", "II"):
        for row in load_dataset(f"MathArena/aime_2024_{part}", split="train"):
            aime.append({"trace_id": f"aime24_{part}_{int(row['problem_idx']):02d}",
                         "dataset": "aime", "prompt": MATH_TEMPLATE.format(problem=row["problem"])})
    aime.sort(key=lambda r: r["trace_id"])

    path = hf_hub_download("livecodebench/code_generation_lite", LCB_FILE, repo_type="dataset")
    rows = [json.loads(line) for line in open(path)]
    hard = sorted((r for r in rows if r["difficulty"] == "hard"), key=lambda r: r["question_id"])
    lcb = [{"trace_id": f"lcb_{r['question_id']}", "dataset": "lcb", "prompt": format_lcb(r)}
           for r in random.Random(SEED).sample(hard, N_LCB)]
    lcb.sort(key=lambda r: r["trace_id"])

    out = []
    for name, items in (("aime", aime), ("lcb", lcb)):
        dev = set(random.Random(SEED).sample(range(len(items)), N_DEV[name]))
        for i, item in enumerate(items):
            out.append({**item, "split": "dev" if i in dev else "test"})
    return out


def build_input_ids(tokenizer, prompt):
    """Qwen3 default chat template, thinking ON (DFlash benchmark.apply_chat_template path)."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def loop_flag(output_ids, window=32):
    """True if >50% of the 32-token windows starting in the second half already appeared earlier."""
    n = len(output_ids)
    if n < 2 * window:
        return False
    seen = set()
    half = n // 2
    repeated = total = 0
    for i in range(n - window + 1):
        w = tuple(output_ids[i:i + window])
        if i >= half:
            total += 1
            repeated += w in seen
        seen.add(w)
    return total > 0 and repeated / total > 0.5
