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


def build_input_ids(tokenizer, prompt, enable_thinking=True):
    """Qwen3 default chat template, thinking ON by default (DFlash benchmark.apply_chat_template path).
    enable_thinking=False only for the Amendment-4 control."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=enable_thinking,
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


def repeat_fraction(output_ids, window=32):
    """Fraction of windows starting in the second half that appeared earlier (the loop_flag statistic)."""
    n = len(output_ids)
    if n < 2 * window:
        return 0.0
    seen, half, rep, tot = set(), n // 2, 0, 0
    for i in range(n - window + 1):
        w = tuple(output_ids[i:i + window])
        if i >= half:
            tot += 1
            rep += w in seen
        seen.add(w)
    return rep / tot if tot else 0.0


def loop_cut(output_ids, window=32, onset_thr=0.8, min_back=512, max_frac=0.35, step=512, min_keep=1024,
             onset="window", onset_win=1000):
    """Amendment 2 (KILL_TEST_BRIEF.md §11) truncation rule for loop-flagged traces.

    r_j = 1 if the 32-token window ending at output position j occurred earlier (j >= 31; else 0).
    p*  = smallest p with mean(r_p..r_{L-1}) >= 0.8.
    P   = distance from the window ending at p* back to its most recent earlier occurrence (if the window at p*
          itself is not a repeat, the first repeated window after p* is used).
    c   = p* - max(512, P); while repeat_fraction(prefix[:c]) > 0.35: c -= 512.  Keep output tokens [0, c).
    Dropped if c < 1024.
    """
    L = len(output_ids)
    last = {}
    r = [0] * L
    back = [0] * L
    for j in range(window - 1, L):
        w = tuple(output_ids[j - window + 1:j + 1])
        if w in last:
            r[j], back[j] = 1, j - last[w]
        last[w] = j
    p_star = None
    if onset == "suffix":  # as specified: smallest p with mean(r_p..r_{L-1}) >= thr
        suffix = 0
        for p in range(L - 1, -1, -1):
            suffix += r[p]
            if suffix / (L - p) >= onset_thr:
                p_star = p
    else:  # proposed: smallest p such that every onset_win-window starting at q >= p has mean >= thr
        pre = [0]
        for x in r:
            pre.append(pre[-1] + x)
        ok_from = L
        for q in range(L - 1, -1, -1):
            hi = min(L, q + onset_win)
            if (pre[hi] - pre[q]) / (hi - q) >= onset_thr:
                ok_from = q
            else:
                break
        p_star = ok_from if ok_from < L else None
    if p_star is None:  # flagged but no onset under this definition: start from the full trace, apply step 5
        period, c = None, L
    else:
        q = next(j for j in range(p_star, L) if r[j])
        period = back[q]
        c = p_star - max(min_back, period)
    while c >= min_keep and repeat_fraction(output_ids[:c]) > max_frac:
        c -= step
    dropped = c < min_keep
    return {"p_star": p_star, "period": period, "c": c, "kept": max(c, 0) if not dropped else 0,
            "frac_kept": repeat_fraction(output_ids[:max(c, 0)]), "dropped": dropped}
