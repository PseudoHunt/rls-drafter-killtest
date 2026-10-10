"""Stage 1 (env-vllm): greedy, thinking-on traces for all 62 problems. Resumable: traces already in
the output file are skipped; new ones are appended after the batch finishes."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.data import build_input_ids, load_problems, loop_flag  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-thinking", action="store_true", help="Amendment-4 control (thinking OFF)")
    ap.add_argument("--only-split", default=None)
    ap.add_argument("--only-dataset", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    problems = [p for p in load_problems()
                if (args.only_split in (None, p["split"])) and (args.only_dataset in (None, p["dataset"]))][: args.limit]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(line)["trace_id"] for line in open(args.out)}
    todo = [p for p in problems if p["trace_id"] not in done]
    print(f"{len(problems)} problems, {len(done)} done, {len(todo)} to generate", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    eos = [151645, 151643]  # Qwen3-4B generation_config eos_token_id (same list DFlash uses as stop ids)
    prompts = [build_input_ids(tok, p["prompt"], enable_thinking=not args.no_thinking) for p in todo]
    max_len = max(len(x) for x in prompts) + args.max_new_tokens + 16
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=0.85,
              seed=0, enable_prefix_caching=False)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, stop_token_ids=eos, seed=0)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=x) for x in prompts], sp)
    gen_s = time.time() - t0
    with open(args.out, "a") as f:
        for p, x, o in zip(todo, prompts, outs):
            c = o.outputs[0]
            out_ids = list(c.token_ids)
            if c.finish_reason == "stop" and (not out_ids or out_ids[-1] not in eos):
                out_ids.append(int(c.stop_reason) if isinstance(c.stop_reason, int) else eos[0])
            f.write(json.dumps({**{k: p[k] for k in ("trace_id", "dataset", "split")},
                                "prompt_len": len(x), "prompt_ids": x, "output_ids": out_ids,
                                "finish_reason": c.finish_reason, "loop": loop_flag(out_ids),
                                "thinking": not args.no_thinking}) + "\n")
    print(f"generated {len(todo)} traces in {gen_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
