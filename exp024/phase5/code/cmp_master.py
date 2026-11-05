#!/usr/bin/env python3
"""EXP-024 第三阶段：fp32 主权重层面比对。M_H = head 主权重，A = 线上候选主权重，B = 本地候选主权重（train_step 存的单文件）。
每个张量：A 与 B 不同的元素数、最大绝对差、相对差范数 ‖B−A‖/‖A−M_H‖、更新 cos(A−M_H, B−M_H)。
  cmp_master.py --head-master DIR --a DIR --b FILE_OR_DIR --out x.json
"""
import argparse, json, math
from pathlib import Path
import torch
from safetensors import safe_open


def index(p):
    p = Path(p); files = [p] if p.is_file() else sorted(p.glob('*.safetensors'))
    out = {}
    for f in files:
        with safe_open(str(f), 'pt') as h:
            for n in h.keys(): out[n] = str(f)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--head-master', required=True); ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    H, A, B = index(a.head_master), index(a.a), index(a.b)
    names = sorted(set(H) & set(A) & set(B)); print('tensors', len(H), len(A), len(B), 'common', len(names), flush=True)
    hd = {}
    def get(path, n):
        if path not in hd: hd[path] = safe_open(path, 'pt')
        return hd[path].get_tensor(n).float().flatten()
    tot = dict(elements=0, a_ne_b=0, max_abs=0.0, dot=0.0, na=0.0, nb=0.0, nab=0.0, a_eq_h=0, b_eq_h=0); per = []
    for n in names:
        h, x, y = get(H[n], n), get(A[n], n), get(B[n], n)
        da, db = x - h, y - h; ne = x != y
        d = dict(name=n, elements=h.numel(), a_ne_b=int(ne.sum()), max_abs=float((x - y).abs().max()), na=float(da.norm()), nb=float(db.norm()),
                 nab=float((x - y).norm()), cos=float(da @ db / (da.norm() * db.norm())) if da.norm() > 0 and db.norm() > 0 else None)
        per.append(d)
        for k in ('elements', 'a_ne_b'): tot[k] += d[k]
        tot['max_abs'] = max(tot['max_abs'], d['max_abs']); tot['dot'] += float(da @ db)
        tot['na'] += d['na'] ** 2; tot['nb'] += d['nb'] ** 2; tot['nab'] += d['nab'] ** 2
    tot['cos'] = tot['dot'] / math.sqrt(tot['na'] * tot['nb']); tot['rel_diff'] = math.sqrt(tot['nab'] / tot['na'])
    for k in ('na', 'nb', 'nab'): tot[k] = math.sqrt(tot[k])
    Path(a.out).write_text(json.dumps(dict(total=tot, per_tensor=per), indent=1)); print(json.dumps(tot))


if __name__ == '__main__':
    main()
