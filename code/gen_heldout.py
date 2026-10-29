#!/usr/bin/env python3
"""EXP-022 (B)：held-out 750 题 × 4 次，截断采样（T=1/top-k 20/top-p 0.95）vs 全词表采样，vLLM。

prompt：官方格式（本机预先分词，见 heldout.json），模型从 <think> 开始生成；max_tokens 8192；EOS = 248046 / 248044。
判分（近似官方 completed-boxed）：以 EOS 结束、有 </think>、</think> 之后最后一个 \\boxed{} 用 math-verify 对标准答案。
输出：OUT/<arm>.jsonl（每个样本：index, j, n_tokens, finish, think_closed, boxed, correct）、OUT/<arm>_capped.jsonl（打满全文，最多 300 条）、
     OUT/summary.json。
用法：python gen_heldout.py --ckpt /data/models/41a1afd0 --tasks heldout.json --out /data/results/B [--arms trunc,full] [--k 4]
"""
import argparse
import json
import re
import time
from pathlib import Path

EOS = [248046, 248044]
ARMS = {'trunc': dict(top_k=20, top_p=0.95), 'full': dict(top_k=-1, top_p=1.0)}


def last_boxed(s):
    i = s.rfind('\\boxed{')
    if i < 0:
        return None
    j, depth = i + len('\\boxed{'), 1
    while j < len(s) and depth:
        depth += {'{': 1, '}': -1}.get(s[j], 0)
        j += 1
    return s[i + len('\\boxed{'):j - 1] if depth == 0 else None


def grade(text, finish, answer):
    from math_verify import parse, verify
    closed = '</think>' in text
    if finish != 'stop' or not closed:
        return closed, None, False
    box = last_boxed(text.split('</think>')[-1])
    if box is None:
        return closed, None, False
    try:
        ok = bool(verify(parse('$' + answer + '$'), parse('$' + box + '$')))
    except Exception:
        ok = False
    return closed, box, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True), ap.add_argument('--tasks', required=True), ap.add_argument('--out', required=True)
    ap.add_argument('--arms', default='trunc,full'), ap.add_argument('--k', type=int, default=4), ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = json.load(open(args.tasks))
    if args.limit:
        tasks = tasks[:args.limit]
    llm = LLM(model=args.ckpt, dtype='bfloat16', max_model_len=12288, gpu_memory_utilization=0.88, seed=0,
              limit_mm_per_prompt={'image': 0, 'video': 0}, enable_prefix_caching=True)
    summary = {}
    for arm in args.arms.split(','):
        t0 = time.time()
        prompts = [TokensPrompt(prompt_token_ids=t['prompt_ids']) for t in tasks]
        params = [SamplingParams(n=args.k, temperature=1.0, max_tokens=8192, stop_token_ids=EOS, skip_special_tokens=False,
                                 seed=20261029 + t['index'], **ARMS[arm]) for t in tasks]
        res = llm.generate(prompts, params, use_tqdm=False)
        rows, capped = [], []
        for t, r in zip(tasks, res):
            for j, o in enumerate(r.outputs):
                closed, box, ok = grade(o.text, o.finish_reason, t['answer'])
                rows.append(dict(index=t['index'], j=j, n_tokens=len(o.token_ids), finish=o.finish_reason, think_closed=closed,
                                 boxed=box, correct=ok, level=t['level'], subject=t['subject']))
                if o.finish_reason == 'length' and len(capped) < 300:
                    capped.append(dict(index=t['index'], j=j, text=o.text))
        (out / f'{arm}.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
        (out / f'{arm}_capped.jsonl').write_text('\n'.join(json.dumps(x) for x in capped))
        n = len(rows)
        s = dict(samples=n, pass_rate=sum(x['correct'] for x in rows) / n,
                 capped=sum(x['finish'] == 'length' for x in rows) / n,
                 eos_no_answer=sum(x['finish'] == 'stop' and not x['correct'] and x['boxed'] is None for x in rows) / n,
                 median_tokens=sorted(x['n_tokens'] for x in rows)[n // 2], seconds=time.time() - t0)
        summary[arm] = s
        print(arm, json.dumps(s), flush=True)
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('B_DONE', flush=True)


if __name__ == '__main__':
    main()
