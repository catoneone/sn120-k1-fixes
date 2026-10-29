#!/usr/bin/env python3
"""EXP-023 门槛评测（官方式）：固定 1,000 道私有题 × 4 次，截断采样（T=1/top-k 20/top-p 0.95），每题固定种子（head 和候选共用随机数）。
判分与 gen_heldout.py 相同。输出 OUT（jsonl：index, j, correct, capped）和 OUT.summary.json。
用法：python gate_eval.py --ckpt DIR --tasks gate.json --out /data/results/gated_x/gate_s03.jsonl [--k 4]
"""
import argparse
import json
import time
from pathlib import Path

from gen_heldout import EOS, grade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--tasks', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=4); ap.add_argument('--gpu-mem', type=float, default=0.78)
    ap.add_argument('--max-seqs', type=int, default=768)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    t0 = time.time()
    tasks = json.load(open(a.tasks))
    llm = LLM(model=a.ckpt, dtype='bfloat16', max_model_len=12288, gpu_memory_utilization=a.gpu_mem, seed=0,
              limit_mm_per_prompt={'image': 0, 'video': 0}, enable_prefix_caching=True, max_num_seqs=a.max_seqs)
    t1 = time.time()
    prompts = [TokensPrompt(prompt_token_ids=t['prompt_ids']) for t in tasks]
    params = [SamplingParams(n=a.k, temperature=1.0, top_k=20, top_p=0.95, max_tokens=8192, stop_token_ids=EOS,
                             skip_special_tokens=False, seed=20261101 + t['index']) for t in tasks]
    res = llm.generate(prompts, params, use_tqdm=False)
    rows = []
    for t, r in zip(tasks, res):
        for j, o in enumerate(r.outputs):
            _, _, ok = grade(o.text, o.finish_reason, t['answer'])
            rows.append(dict(index=t['index'], j=j, correct=bool(ok), capped=o.finish_reason == 'length', n=len(o.token_ids)))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(json.dumps(x) for x in rows))
    n = len(rows)
    s = dict(samples=n, pass_rate=sum(x['correct'] for x in rows) / n, capped=sum(x['capped'] for x in rows) / n,
             load_s=t1 - t0, gen_s=time.time() - t1)
    Path(str(out) + '.summary.json').write_text(json.dumps(s))
    print('GATE_EVAL', json.dumps(s), flush=True)


if __name__ == '__main__':
    main()
