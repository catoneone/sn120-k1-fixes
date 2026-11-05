#!/usr/bin/env python3
"""EXP-024 第五阶段 R3：各模式梯度两两比较，逐张量 cos / 范数比，并按层段 × 类型汇总。
  cmp_grads.py --dir /dev/shm/grads --pairs kit:fp32ref live:fp32ref kit:live live_fusednorm:fp32ref live_fusednorm:live --out cmp.json"""
import argparse, collections, json, math, re
from safetensors import safe_open


def cls(name):
    if 'linear_attn' in name:
        if re.search(r'A_log|dt_bias|in_proj_a|in_proj_b', name): return 'gdn_gate'
        if 'norm' in name: return 'gdn_norm'
        if 'conv1d' in name: return 'gdn_conv'
        return 'gdn_other'
    if 'self_attn' in name: return 'attn'
    if 'mlp' in name: return 'mlp'
    return 'other'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--pairs', nargs='+', required=True)
    ap.add_argument('--out', required=True); a = ap.parse_args()
    out = {}
    for pr in a.pairs:
        x, y = pr.split(':')
        A = safe_open(f'{a.dir}/grads_{x}.safetensors', 'pt'); B = safe_open(f'{a.dir}/grads_{y}.safetensors', 'pt')
        per = []; grp = collections.defaultdict(lambda: [0.0, 0.0, 0.0]); tot = [0.0, 0.0, 0.0]
        for n in sorted(set(A.keys()) & set(B.keys())):
            ga, gb = A.get_tensor(n).double().flatten(), B.get_tensor(n).double().flatten()
            d, na, nb = float(ga @ gb), float(ga @ ga), float(gb @ gb)
            per.append(dict(name=n, cos=d / math.sqrt(na * nb) if na and nb else None, ratio=math.sqrt(na / nb) if nb else None))
            m = re.match(r'model\.language_model\.layers\.(\d+)\.', n)
            seg = '%02d-%02d' % (int(m.group(1)) // 8 * 8, int(m.group(1)) // 8 * 8 + 7) if m else 'top'
            for k in ((seg, cls(n)), (seg, 'all')):
                g = grp[k]; g[0] += d; g[1] += na; g[2] += nb
            tot[0] += d; tot[1] += na; tot[2] += nb
        summ = {f'{s}|{c}': dict(cos=g[0] / math.sqrt(g[1] * g[2]) if g[1] and g[2] else None, ratio=math.sqrt(g[1] / g[2]) if g[2] else None)
                for (s, c), g in sorted(grp.items())}
        out[pr] = dict(total=dict(cos=tot[0] / math.sqrt(tot[1] * tot[2]), ratio=math.sqrt(tot[1] / tot[2])), groups=summ, per_tensor=per)
        print(pr, json.dumps(out[pr]['total']), flush=True)
        for k, v in summ.items():
            if k.endswith('|all') or 'gdn_gate' in k: print('   ', k, {kk: round(vv, 4) for kk, vv in v.items() if vv is not None})
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
