#!/usr/bin/env python3
"""EXP-024：按线上门槛的方式在一轮揭晓的抽题上采样 + 判分（kit 的 sampler / grade / baseline_v1 原样调用）。

  evalrun.py --url http://127.0.0.1:8000 --tok /data/models/H --round round600.json --role head|cand \
             --seed prod|<任意字符串> [--epoch 600] --out out.json

role=head：3 个 plan seed × 4 次（BV.plan_seed(S, b)）；role=cand：1 个 plan seed × 8 次（BV.cand_seed(S, epoch)）。
S = 揭晓文件里的 seed（--seed prod）或 sha256('exp024:'+字符串)（新种子）。请求种子 = sampler.request_seed(P, 'affine_gatepool',
probe_index, j)，和 gate_eval.py / 线上 probe 一样。输出逐题格式同线上 green_turn.*_per_task（n c w u b x tok L），另存逐条记录。
"""
import argparse, asyncio, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

KIT = Path(os.environ.get('KIT', '/data/affine/miner/kit'))
sys.path[:0] = [str(KIT), str(KIT.parent / 'validator' / 'gate')]
import sampler, grade  # noqa: E402
import baseline_v1 as BV  # noqa: E402

ENV = 'affine_gatepool'
MANDATE = ("Solve the math problem. Reason step by step in plain text, then end your response with the final answer in `\\boxed{}`. "
           "Emit exactly one `\\boxed{}` block and nothing after it.")   # = prompts.MANDATE = grade.SYSTEM


def gate_ids(tok, problem):   # = prompts.gate_ids
    ids = tok.apply_chat_template([dict(role='system', content=MANDATE), dict(role='user', content=problem)], tokenize=True,
                                  return_dict=False, add_generation_prompt=True, enable_thinking=True)
    return list(ids['input_ids'] if isinstance(ids, dict) else ids)


def _ginit(tokdir):
    grade.init(tokdir)


def _grade(args):
    out, stop, gold, problem = args
    return grade.grade(out, stop, gold, problem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', action='append', required=True); ap.add_argument('--tok', required=True)
    ap.add_argument('--round', required=True); ap.add_argument('--role', choices=('head', 'cand'), required=True)
    ap.add_argument('--seed', default='prod'); ap.add_argument('--epoch', type=int)
    ap.add_argument('--out', required=True); ap.add_argument('--concurrency', type=int, default=512)
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 题（冒烟用）')
    a = ap.parse_args()
    R = json.loads(Path(a.round).read_text())
    G = json.loads((KIT.parent / 'validator' / 'gate' / 'gate-config.public.json').read_text())
    cfg = dict(BV.DEFAULTS, **G['heldout']['prior_v1'])
    S = R['seed'] if a.seed == 'prod' else hashlib.sha256(('exp024:' + a.seed).encode()).hexdigest()
    epoch = a.epoch or R['green_turn']['epoch']
    if a.role == 'head':
        nb = int(cfg['head_blocks']); kb = int(cfg['head_rollouts']) // nb
        plans = [(BV.plan_seed(S, b), kb) for b in range(nb)]
        if a.seed == 'prod': assert [p for p, _ in plans] == R['green_turn']['head_plan_seeds'], 'head plan seed mismatch'
    else:
        plans = [(BV.cand_seed(S, epoch), int(cfg['k_cand']))]
        if a.seed == 'prod' and epoch == R['green_turn']['epoch']: assert plans[0][0] == R['green_turn']['plan_seed'], 'cand plan seed mismatch'
    tasks = {t['id']: t for t in R['tasks']}
    ids_order = list(R['drawn_ids'])[:a.limit or None]
    grade.init(a.tok); tok = grade._G['tok']
    prompt = {t: gate_ids(tok, tasks[t]['problem']) for t in ids_order}
    jobs = []
    for bi, (P, k) in enumerate(plans):
        for t in ids_order:
            for j in range(k):
                jobs.append(((bi, t, j), prompt[t], sampler.request_seed(P, ENV, BV.probe_index(t, cfg), j)))
    t0 = time.time()
    print(f'{a.out}: {len(jobs)} requests, {len(ids_order)} tasks, plans {[p[:8] for p, _ in plans]}', flush=True)
    res = asyncio.run(sampler.sample_all(a.url, jobs, concurrency=a.concurrency))
    t1 = time.time()
    keys = [j[0] for j in jobs]
    with ProcessPoolExecutor(max_workers=min(48, os.cpu_count() or 8), initializer=_ginit, initargs=(a.tok,)) as ex:
        kinds = list(ex.map(_grade, [(res[k][0], res[k][1], tasks[k[1]]['answer'], tasks[k[1]]['problem']) for k in keys], chunksize=16))
    cells, recs, ntok = {}, [], 0
    for k, (kind, how) in zip(keys, kinds):
        bi, t, j = k; o, stop = res[k]
        pt = cells.setdefault(t, dict(n=0, c=0, w=0, u=0, b=0, x=0, tok=0, L=[]))
        pt['n'] += 1; pt['tok'] += len(o); pt['L'].append(len(o)); pt['b'] += stop == 'budget'
        pt[{'positive': 'c', 'negative': 'w', 'neutral': 'u'}.get(kind, 'x')] += 1
        recs.append([bi, t, j, len(o), stop, kind, how]); ntok += len(o)
    doc = dict(round=R['round_id'], role=a.role, seed=a.seed, S=S, epoch=epoch, plans=[p for p, _ in plans], tok_dir=a.tok,
               sample_s=t1 - t0, grade_s=time.time() - t1, tokens=ntok, tok_per_s=ntok / max(1e-9, t1 - t0), per_task=cells, rollouts=recs)
    Path(a.out).write_text(json.dumps(doc))
    c = sum(v['c'] for v in cells.values()); n = sum(v['n'] for v in cells.values()); b = sum(v['b'] for v in cells.values())
    print(f'DONE {a.out}: pass {c / n:.4f} cap {b / n:.4f} tokens {ntok / 1e6:.1f}M in {t1 - t0:.0f}s ({ntok / max(1, t1 - t0):.0f} tok/s), grade {time.time() - t1:.0f}s', flush=True)


if __name__ == '__main__':
    main()
