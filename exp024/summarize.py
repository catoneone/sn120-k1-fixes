#!/usr/bin/env python3
"""EXP-024 汇总：results/ 下的本地评测 vs 线上揭晓的逐题成绩（round 600 / 607），以及权重、训练报告比对。

python3 experiments/EXP-024-gate-variance-local-repro/summarize.py [results 目录]
"""
import importlib.util, json, math, os, sys
from pathlib import Path

E = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else E / 'results'
spec = importlib.util.spec_from_file_location('BV', E / 'stage' / 'baseline_v1.py'); BV = importlib.util.module_from_spec(spec); spec.loader.exec_module(BV)
POOL = {'affine_math': 7492, 'dapo': 11813}
env_of = lambda t: 'dapo' if t.startswith('dapo') else 'affine_math'
rate = lambda x: x['c'] / x['n']

rounds = {k: json.loads((E / 'stage' / 'data' / f'round{k}.json').read_text()) for k in ('600', '607')}
runs = {}
for k, r in rounds.items():
    runs[f'prod{k}:head'] = dict(per_task=r['green_turn']['head_per_task'], round=k)
    runs[f'prod{k}:cand'] = dict(per_task=r['green_turn']['candidate_per_task'], round=k)
for f in sorted(RES.glob('*.json')):
    if f.name.startswith(('H_', 'C0_', 'C1_', 'C2c_', 'C4_')):
        d = json.loads(f.read_text()); d['round'] = '607' if 'D607' in f.name else '600'; runs[f.stem] = d


def summary(name):
    P = runs[name]['per_task']; n = sum(v['n'] for v in P.values())
    c = sum(v['c'] for v in P.values()); b = sum(v['b'] for v in P.values()); tok = sum(v['tok'] for v in P.values())
    return dict(run=name, tasks=len(P), rollouts=n, pass_=c / n, cap=b / n, mean_len=tok / n)


def z_of(hname, cname, rk):
    R = rounds[rk]; pi0 = R['settings']['N'] / sum(POOL.values())
    drawn = {t: dict(env=env_of(t), pi=pi0) for t in R['drawn_ids']}
    ev = BV.estimate(drawn, runs[hname]['per_task'], runs[cname]['per_task'], POOL)['evaluated']
    return dict(head=hname, cand=cname, delta=ev['delta'], se=ev['se'], z=ev['z'], n=ev['n'],
                **{f'gain_{e}': v['gain'] for e, v in ev['per_env'].items()})


def paired(a, b):
    """同一模型 / 同类运行的逐题比较：平均差（按题等权）、离散度（Σ(x−y)²/Var，二项期望下 ≈ 1）、长度逐条相同的比例。"""
    A, B = runs[a]['per_task'], runs[b]['per_task']
    ts = [t for t in A if t in B]
    d = [rate(A[t]) - rate(B[t]) for t in ts]
    m = sum(d) / len(d)
    chi = 0.0; df = 0
    for t in ts:
        na, nb = A[t]['n'], B[t]['n']; p = (A[t]['c'] + B[t]['c']) / (na + nb)
        v = p * (1 - p) * (1 / na + 1 / nb)
        if v > 0: chi += (rate(A[t]) - rate(B[t])) ** 2 / v; df += 1
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
    same_len = tot_len = 0
    for t in ts:
        La, Lb = A[t].get('L') or [], B[t].get('L') or []
        if len(La) == len(Lb):
            tot_len += len(La); same_len += sum(x == y for x, y in zip(La, Lb))
    la = sum(A[t]['tok'] for t in ts) / sum(A[t]['n'] for t in ts); lb = sum(B[t]['tok'] for t in ts) / sum(B[t]['n'] for t in ts)
    ca = sum(A[t]['b'] for t in ts) / sum(A[t]['n'] for t in ts); cb = sum(B[t]['b'] for t in ts) / sum(B[t]['n'] for t in ts)
    return dict(a=a, b=b, tasks=len(ts), mean_diff_pp=100 * m, se_pp=100 * se, z=m / se if se else None, dispersion=chi / df if df else None,
                len_exact_match=same_len / tot_len if tot_len else None, mean_len_a=la, mean_len_b=lb, cap_a=ca, cap_b=cb)


out = dict(runs=[summary(k) for k in runs])
pairs = [('H_Dp_prod', 'prod600:head'), ('C1_Dp_prod', 'prod600:cand'), ('H_Dp_s1', 'H_Dp_prod'), ('H_Dp_s1', 'prod600:head'),
         ('C1_Dp_s1', 'C1_Dp_prod'), ('C1_Dp_s1', 'prod600:cand'), ('C2c_Dp_prod', 'C1_Dp_prod'), ('C2c_Dp_prod', 'prod600:cand'),
         ('C0_Dp_s1', 'H_Dp_prod')]
out['paired'] = [paired(a, b) for a, b in pairs if a in runs and b in runs]
heads600 = [k for k in ('prod600:head', 'H_Dp_prod', 'H_Dp_s1') if k in runs]
cands600 = ['prod600:cand'] + sorted(k for k in runs if k.startswith(('C0_Dp', 'C1_Dp', 'C2c_Dp')))
out['z600'] = [z_of(h, c, '600') for h in heads600 for c in cands600]
if 'H_Dp_s1' in runs:
    out['z600'] += [z_of('H_Dp_s1', 'H_Dp_prod', '600'), z_of('H_Dp_prod', 'prod600:head', '600')]
def grid(prefix_list):
    zs = [z_of(h, c, '600') for h in heads600 for c in cands600 if c in prefix_list]
    v = [r['z'] for r in zs]; d = [r['delta'] for r in zs]
    m = sum(v) / len(v); sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else None
    return dict(cands=prefix_list, n=len(v), z_mean=m, z_sd=sd, z_min=min(v), z_max=max(v), p_z_gt_1=sum(x > 1 for x in v) / len(v),
                delta_mean_pp=100 * sum(d) / len(d))
def cand_rates(prefix_list):   # 同一权重不同种子：候选通过率（按题等权、按环境份额加权）的离散
    vals = []
    for c in prefix_list:
        P = runs[c]['per_task']; acc = {}
        for t, v in P.items(): acc.setdefault(env_of(t), []).append(rate(v))
        vals.append(sum(POOL[e] / sum(POOL.values()) * sum(x) / len(x) for e, x in acc.items()))
    m = sum(vals) / len(vals)
    return dict(cands=prefix_list, rates=vals, sd_pp=100 * math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else None)
C1s = [c for c in cands600 if c.startswith(('prod600:cand', 'C1_'))]
C2s = [c for c in cands600 if c.startswith('C2c_')]
C0s = [c for c in cands600 if c.startswith('C0_')]
out['grid'] = [grid(x) for x in (C1s, C2s, C0s) if x]
out['cand_rates'] = [cand_rates(x) for x in (C1s, C2s) if len(x) > 1]
if 'C4_D607_prod' in runs:
    out['z607'] = [z_of('prod607:head', 'C4_D607_prod', '607'), z_of('prod607:head', 'prod607:cand', '607')]
for f in ('weights_C1_vs_C2c.json',):
    if (RES / f).exists(): out['weights'] = json.loads((RES / f).read_text())['total']
turn = json.loads((E / 'stage' / 'data' / 'turn600.json').read_text())['training']
keys = ('objective_loss', 'preclip_gradient_norm', 'rollout_count', 'trained_rollouts', 'loss_tokens')
for tag in ('C2c', 'C4'):
    p = RES / f'step_{tag}.json'
    if p.exists():
        rep = json.loads(p.read_text())['report']
        out[f'step_{tag}'] = dict({k: (rep.get(k), turn.get(k)) for k in keys},
                                  outside_tokens=((rep.get('support_renorm') or {}).get('outside_tokens'), turn['support_renorm']['outside_tokens']),
                                  projected_changed=(((rep.get('optimizer') or {}).get('sgd') or {}).get('projected_changed_elements'),
                                                     turn['optimizer'].get('projected_changed_elements')))
(RES / 'summary.json').write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print('==', k)
    for row in (v if isinstance(v, list) else [v]):
        print('  ', json.dumps({a: (round(b, 5) if isinstance(b, float) else b) for a, b in row.items()}, ensure_ascii=False))
