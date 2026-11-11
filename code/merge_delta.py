#!/usr/bin/env python3
"""Tools for EXPERIMENT-bigfile.md: work on fp32 masters of candidates that were all trained from the same head.

A "master" is a directory of *.safetensors shards (fetch_fp32_master.py), a single .safetensors file, or a candidate
directory written by kit/train_step.py (its fp32-master/ subdirectory is used). d_k = master(candidate k) - master(head).

  cos    pairwise cosines of d_1..d_K (overall and per block): how much independent files agree on the direction.
         python merge_delta.py cos --head-master HM --cands S1 S2 ... S8 --out delta_cos.json

  merge  build the candidate  master(head) + scale * sum_k d_k  and its bf16 checkpoint (round-to-nearest-even of the
         new master for the trainable tensors; everything else copied from the bf16 head directory).
         python merge_delta.py merge --head-master HM --cands S1 ... S8 --scale 0.35355339 --head HEAD_DIR --out B1_DIR

  chain  check a chain C_1..C_K (C_k trained from C_{k-1}, C_0 = head) against the single steps S_1..S_K:
         step k of the chain, u_k = master(C_k) - master(C_{k-1}), vs scale * d_k (cosine, norm ratio), and the whole chain
         vs scale * sum_k d_k.
         python merge_delta.py chain --head-master HM --singles S1 ... S8 --chain C1 ... C8 --scale 0.35355339 --out chain_check.json

Dots are accumulated in float64. Peak memory is about (K + 2) x the largest tensor in fp32 (cos / merge), plus the whole
new master for merge (it is written as one file, like train_step.py does).
"""
import argparse, json, math, re, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

CHUNK = 1 << 24


def index(p):
    """{tensor name: file} for a shard directory, a single file, or a train_step.py candidate directory."""
    p = Path(p)
    if p.is_dir() and (p / 'fp32-master').is_dir():
        p = p / 'fp32-master'
    files = [p] if p.is_file() else sorted(p.glob('*.safetensors'))
    if not files:
        raise SystemExit(f'no .safetensors under {p}')
    out = {}
    for f in files:
        with safe_open(str(f), 'pt') as h:
            for n in h.keys():
                out[n] = str(f)
    return out


class Reader:
    def __init__(self):
        self.h = {}

    def get(self, path, name):
        if path not in self.h:
            self.h[path] = safe_open(path, 'pt')
        t = self.h[path].get_tensor(name)
        if t.dtype != torch.float32:
            raise SystemExit(f'{name} in {path} is {t.dtype}, expected float32 (is this an fp32 master?)')
        return t.flatten()

    def shape(self, path, name):
        if path not in self.h:
            self.h[path] = safe_open(path, 'pt')
        return self.h[path].get_slice(name).get_shape()


def dot(a, b):
    s = 0.0
    for i in range(0, a.numel(), CHUNK):
        s += float(a[i:i + CHUNK].double() @ b[i:i + CHUNK].double())
    return s


def block(name):
    """Coarse block of a tensor: embed, lm_head, layers 0-7 / 8-15 / 16-23 / 24-31 split into attn and mlp, other."""
    if 'embed_tokens' in name:
        return 'embed'
    if name.startswith('lm_head'):
        return 'lm_head'
    m = re.search(r'layers\.(\d+)\.', name)
    if not m:
        return 'other'
    lo = int(m.group(1)) // 8 * 8
    kind = 'attn' if 'attn' in name else 'mlp' if 'mlp' in name else 'norm'
    return f'layers{lo}-{lo + 7}.{kind}'


def common_names(*idx):
    names = set(idx[0])
    for i in idx[1:]:
        if set(i) != names:
            miss = sorted(names ^ set(i))[:5]
            raise SystemExit(f'tensor names differ between masters, e.g. {miss}')
    return sorted(names)


def gram_summary(G, K):
    norms = [math.sqrt(G[i][i]) for i in range(K)]
    cos = [[G[i][j] / (norms[i] * norms[j]) if norms[i] > 0 and norms[j] > 0 else None for j in range(K)] for i in range(K)]
    off = [cos[i][j] for i in range(K) for j in range(i + 1, K) if cos[i][j] is not None]
    c = sum(off) / len(off) if off else None
    tot = sum(G[i][j] for i in range(K) for j in range(K))
    out = dict(norms=norms, cos=cos, mean_offdiag_cos=c, min_offdiag_cos=min(off) if off else None, max_offdiag_cos=max(off) if off else None,
               sum_norm=math.sqrt(max(tot, 0.0)), sum_norm_over_sqrtK_mean_norm=math.sqrt(max(tot, 0.0)) / (math.sqrt(K) * sum(norms) / K) if sum(norms) > 0 else None)
    if c is not None and c > 0:
        # if every d_k = signal + independent noise of the same size, cos between two of them = signal share c;
        # the sum of N of them then has signal share N c / (1 + (N - 1) c)
        out['signal_share_of_sum'] = {str(n): n * c / (1 + (n - 1) * c) for n in (1, 2, 4, 8, 16, 32, 64)}
    return out


def cmd_cos(a):
    H = index(a.head_master); C = [index(c) for c in a.cands]; K = len(C)
    names = common_names(H, *C); rd = Reader()
    G = [[0.0] * K for _ in range(K)]; GB = {}
    for n in names:
        h = rd.get(H[n], n); d = [rd.get(C[k][n], n) - h for k in range(K)]
        b = GB.setdefault(block(n), [[0.0] * K for _ in range(K)])
        for i in range(K):
            for j in range(i, K):
                v = dot(d[i], d[j])
                G[i][j] += v; b[i][j] += v
                if j != i:
                    G[j][i] += v; b[j][i] += v
    out = dict(K=K, tensors=len(names), cands=a.cands, total=gram_summary(G, K), blocks={k: gram_summary(v, K) for k, v in sorted(GB.items())})
    Path(a.out).write_text(json.dumps(out, indent=1))
    t = out['total']
    print(json.dumps(dict(K=K, tensors=len(names), mean_offdiag_cos=t['mean_offdiag_cos'], min=t['min_offdiag_cos'], max=t['max_offdiag_cos'],
                          sum_norm_over_sqrtK_mean_norm=t['sum_norm_over_sqrtK_mean_norm'])))


def cmd_merge(a):
    H = index(a.head_master); C = [index(c) for c in a.cands]; K = len(C)
    names = common_names(H, *C); rd = Reader(); head, out = Path(a.head), Path(a.out)
    idx = json.loads((head / 'model.safetensors.index.json').read_text())
    missing = sorted(set(names) - set(idx['weight_map']))
    if missing:
        raise SystemExit(f'{len(missing)} master tensors are not in the head checkpoint index, e.g. {missing[:5]}')
    out.mkdir(parents=True, exist_ok=True); (out / 'fp32-master').mkdir(exist_ok=True)
    for p in head.iterdir():                               # config / tokenizer / chat template: copied as is
        if p.is_file() and p.suffix != '.safetensors' and p.name != 'model.safetensors.index.json':
            shutil.copy2(p, out / p.name)
    new, upd2, h2 = {}, 0.0, 0.0
    for n in names:
        h = rd.get(H[n], n); s = torch.zeros(h.numel(), dtype=torch.float64)
        for k in range(K):
            s += (rd.get(C[k][n], n) - h).double()
        u = a.scale * s
        upd2 += float(u @ u); h2 += dot(h, h)
        shape = rd.shape(H[n], n)
        new[n] = (h.double() + u).float().reshape(shape)
    save_file(new, str(out / 'fp32-master' / 'fp32-master.safetensors'))
    shards = {}
    for k, f in idx['weight_map'].items():
        shards.setdefault(f, []).append(k)
    replaced, changed = 0, 0
    for f, keys in sorted(shards.items()):
        t = {}
        with safe_open(str(head / f), 'pt') as hh:
            for k in keys:
                t[k] = hh.get_tensor(k)
        for k in keys:
            if k in new:
                x = new[k].to(torch.bfloat16)              # torch fp32 -> bf16 cast is round-to-nearest-even
                if x.shape != t[k].shape or t[k].dtype != torch.bfloat16:
                    raise SystemExit(f'{k}: head {t[k].dtype} {tuple(t[k].shape)} vs master {tuple(x.shape)}')
                changed += int((x != t[k]).sum()); t[k] = x; replaced += 1
        save_file(t, str(out / f), metadata={'format': 'pt'})
    shutil.copy2(head / 'model.safetensors.index.json', out / 'model.safetensors.index.json')
    rep = dict(head=str(head), head_master=a.head_master, cands=a.cands, scale=a.scale, tensors=len(names), replaced=replaced,
               update_norm_master=math.sqrt(upd2), relative_update=math.sqrt(upd2 / h2) if h2 > 0 else None, bf16_elements_changed=changed)
    (out / 'MERGE.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep))


def cmd_chain(a):
    if len(a.singles) != len(a.chain):
        raise SystemExit('--singles and --chain need the same number of entries')
    H = index(a.head_master); S = [index(s) for s in a.singles]; C = [index(c) for c in a.chain]; K = len(S)
    names = common_names(H, *S, *C); rd = Reader()
    acc = [dict(dot=0.0, nu=0.0, nd=0.0, ndiff=0.0) for _ in range(K)]; tot = dict(dot=0.0, nu=0.0, nd=0.0, ndiff=0.0)
    for n in names:
        h = rd.get(H[n], n); prev = h; ssum = torch.zeros(h.numel(), dtype=torch.float64)
        for k in range(K):
            cur = rd.get(C[k][n], n); u = (cur - prev).double(); d = a.scale * (rd.get(S[k][n], n) - h).double(); ssum += d
            x = acc[k]; x['dot'] += float(u @ d); x['nu'] += float(u @ u); x['nd'] += float(d @ d); x['ndiff'] += float((u - d) @ (u - d))
            prev = cur
        u = (prev - h).double()
        tot['dot'] += float(u @ ssum); tot['nu'] += float(u @ u); tot['nd'] += float(ssum @ ssum); tot['ndiff'] += float((u - ssum) @ (u - ssum))

    def fin(x):
        ok = x['nu'] > 0 and x['nd'] > 0
        return dict(cos=x['dot'] / math.sqrt(x['nu'] * x['nd']) if ok else None, norm_chain=math.sqrt(x['nu']), norm_single_scaled=math.sqrt(x['nd']),
                    norm_ratio=math.sqrt(x['nu'] / x['nd']) if ok else None, rel_diff=math.sqrt(x['ndiff'] / x['nd']) if x['nd'] > 0 else None)
    out = dict(K=K, scale=a.scale, steps=[dict(step=k + 1, **fin(acc[k])) for k in range(K)], whole_chain_vs_scaled_sum=fin(tot),
               note='step 1 starts from the head, so it should equal scale * d_1 (cos ~ 1, rel_diff ~ 0) if a lower lr scales the step exactly')
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(step1=out['steps'][0], whole=out['whole_chain_vs_scaled_sum'])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('cos'); p.add_argument('--head-master', required=True); p.add_argument('--cands', nargs='+', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('merge'); p.add_argument('--head-master', required=True); p.add_argument('--cands', nargs='+', required=True)
    p.add_argument('--scale', type=float, required=True); p.add_argument('--head', required=True, help='bf16 head checkpoint directory'); p.add_argument('--out', required=True)
    p = sub.add_parser('chain'); p.add_argument('--head-master', required=True); p.add_argument('--singles', nargs='+', required=True)
    p.add_argument('--chain', nargs='+', required=True); p.add_argument('--scale', type=float, required=True); p.add_argument('--out', required=True)
    a = ap.parse_args()
    dict(cos=cmd_cos, merge=cmd_merge, chain=cmd_chain)[a.cmd](a)


if __name__ == '__main__':
    main()
