#!/usr/bin/env python3
"""EXP-024 E2a：本地训练的候选 vs 线上候选，逐张量比（相对同一个 head）。

  cmp_weights.py --head H --a C1(线上) --b C2c(本地) --out weights_diff.json

每个张量：head→a、head→b 改动的元素数；a 和 b 不同的元素数、最大差（以 a 处 bf16 刻度计）；更新向量 (a−H)、(b−H) 的范数和 cos。
"""
import argparse, json, math
from pathlib import Path
import torch
from safetensors import safe_open


def index(d):
    p = Path(d) / 'model.safetensors.index.json'
    if not p.exists():   # train_step 存成单文件
        with safe_open(str(Path(d) / 'model.safetensors'), 'pt') as f: return {n: str(Path(d) / 'model.safetensors') for n in f.keys()}
    idx = json.loads(p.read_text())['weight_map']
    return {n: str(Path(d) / f) for n, f in idx.items()}


def ulp(x):   # bf16 刻度：2^(floor(log2|x|) - 7)，0 处用最小正规数
    ax = x.abs().clamp_min(2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(ax)) - 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', required=True); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    H, A, B = index(a.head), index(a.a), index(a.b)
    names = sorted(set(H) & set(A) & set(B))
    print('tensors', len(H), len(A), len(B), 'common', len(names), flush=True)
    tot = dict(elements=0, changed_a=0, changed_b=0, changed_both=0, a_ne_b=0, max_ulp=0.0, dot=0.0, na=0.0, nb=0.0, nab=0.0)
    per = []
    handles = {}
    def get(path, n):
        if path not in handles: handles[path] = safe_open(path, 'pt')
        return handles[path].get_tensor(n)
    for n in names:
        h = get(H[n], n); x = get(A[n], n); y = get(B[n], n)
        if h.dtype != torch.bfloat16: continue
        hf, xf, yf = h.float().flatten(), x.float().flatten(), y.float().flatten()
        da, db = xf - hf, yf - hf
        ca, cb = da != 0, db != 0
        ne = xf != yf
        mu = float(((xf - yf).abs() / ulp(xf))[ne].max()) if bool(ne.any()) else 0.0
        d = dict(name=n, elements=hf.numel(), changed_a=int(ca.sum()), changed_b=int(cb.sum()), changed_both=int((ca & cb).sum()),
                 a_ne_b=int(ne.sum()), max_ulp=mu, na=float(da.norm()), nb=float(db.norm()), nab=float((xf - yf).norm()),
                 cos=float((da @ db) / (da.norm() * db.norm())) if da.norm() > 0 and db.norm() > 0 else None)
        per.append(d)
        for k in ('elements', 'changed_a', 'changed_b', 'changed_both', 'a_ne_b'): tot[k] += d[k]
        tot['max_ulp'] = max(tot['max_ulp'], mu); tot['dot'] += float(da @ db); tot['na'] += d['na'] ** 2; tot['nb'] += d['nb'] ** 2; tot['nab'] += d['nab'] ** 2
    tot['cos'] = tot['dot'] / math.sqrt(tot['na'] * tot['nb']) if tot['na'] and tot['nb'] else None
    for k in ('na', 'nb', 'nab'): tot[k] = math.sqrt(tot[k])
    tot['rel_diff'] = tot['nab'] / tot['na'] if tot['na'] else None
    Path(a.out).write_text(json.dumps(dict(head=a.head, a=a.a, b=a.b, total=tot, per_tensor=per), indent=1))
    print(json.dumps(tot))


if __name__ == '__main__':
    main()
