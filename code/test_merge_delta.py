#!/usr/bin/env python3
"""CPU test of merge_delta.py on small synthetic masters (no GPU, no real checkpoint).

  python3 code/test_merge_delta.py

Builds a fake head (bf16 checkpoint in 2 shards + fp32 master in 2 shards, plus one non-trainable tensor), K = 4 single-step
candidates with a known shared component, and a chain. Checks cos against a direct computation, merge against the expected
master and its bf16 rounding, and chain on (a) a chain that equals the scaled single steps and (b) a perturbed one.
"""
import json, math, subprocess, sys, tempfile
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file

HERE = Path(__file__).resolve().parent
NAMES = {'model.language_model.embed_tokens.weight': (50, 8), 'model.language_model.layers.0.self_attn.q_proj.weight': (8, 8),
         'model.language_model.layers.0.mlp.up_proj.weight': (16, 8), 'model.language_model.layers.9.mlp.up_proj.weight': (16, 8),
         'model.language_model.layers.31.linear_attn.k_proj.weight': (8, 8), 'lm_head.weight': (50, 8)}
FROZEN = 'model.visual.patch_embed.weight'
K, SCALE = 4, 1 / math.sqrt(4)


def run(*args):
    r = subprocess.run([sys.executable, str(HERE / 'merge_delta.py'), *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('merge_delta.py failed:\n' + r.stdout + r.stderr)
    return r.stdout


def write_candidate(d, master):
    (d / 'fp32-master').mkdir(parents=True)
    save_file(master, str(d / 'fp32-master' / 'fp32-master.safetensors'))


def main():
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp(prefix='merge_delta_test_'))
    hm = {n: torch.randn(s) for n, s in NAMES.items()}
    names = sorted(hm)
    # head master in 2 shards; bf16 head in 2 shards with one non-trainable tensor
    HM = tmp / 'H-fp32-master'; HM.mkdir()
    save_file({n: hm[n] for n in names[:3]}, str(HM / 'fp32-master-00001-of-00002.safetensors'))
    save_file({n: hm[n] for n in names[3:]}, str(HM / 'fp32-master-00002-of-00002.safetensors'))
    H = tmp / 'H'; H.mkdir()
    frozen = torch.randn(4, 4).to(torch.bfloat16)
    sh1 = {n: hm[n].to(torch.bfloat16) for n in names[:2]}; sh1[FROZEN] = frozen
    sh2 = {n: hm[n].to(torch.bfloat16) for n in names[2:]}
    save_file(sh1, str(H / 'model-00001-of-00002.safetensors')); save_file(sh2, str(H / 'model-00002-of-00002.safetensors'))
    wm = {n: 'model-00001-of-00002.safetensors' for n in sh1}; wm.update({n: 'model-00002-of-00002.safetensors' for n in sh2})
    (H / 'model.safetensors.index.json').write_text(json.dumps(dict(metadata={}, weight_map=wm)))
    (H / 'config.json').write_text('{}')
    # single steps: d_k = shared + noise_k, small relative to the weights
    shared = {n: 0.02 * torch.randn(s) for n, s in NAMES.items()}
    d = [{n: shared[n] + 0.05 * torch.randn(NAMES[n]) for n in names} for _ in range(K)]
    S = []
    for k in range(K):
        p = tmp / f'S{k + 1}'; write_candidate(p, {n: hm[n] + d[k][n] for n in names}); S.append(p)

    # --- cos
    run('cos', '--head-master', HM, '--cands', *S, '--out', tmp / 'cos.json')
    got = json.loads((tmp / 'cos.json').read_text())
    real = [{n: (load_file(str(S[k] / 'fp32-master' / 'fp32-master.safetensors'))[n] - hm[n]).double().flatten() for n in names} for k in range(K)]
    flat = [torch.cat([real[k][n] for n in names]) for k in range(K)]
    for i in range(K):
        for j in range(K):
            want = float(flat[i] @ flat[j] / (flat[i].norm() * flat[j].norm()))
            assert abs(got['total']['cos'][i][j] - want) < 1e-9, (i, j, got['total']['cos'][i][j], want)
    off = [float(flat[i] @ flat[j] / (flat[i].norm() * flat[j].norm())) for i in range(K) for j in range(i + 1, K)]
    assert abs(got['total']['mean_offdiag_cos'] - sum(off) / len(off)) < 1e-9
    assert set(got['blocks']) == {'embed', 'lm_head', 'layers0-7.attn', 'layers0-7.mlp', 'layers8-15.mlp', 'layers24-31.attn'}, sorted(got['blocks'])
    e = [real[k]['model.language_model.embed_tokens.weight'] for k in range(K)]
    assert abs(got['blocks']['embed']['cos'][0][1] - float(e[0] @ e[1] / (e[0].norm() * e[1].norm()))) < 1e-9
    print('cos ok: mean off-diagonal', round(got['total']['mean_offdiag_cos'], 4))

    # --- merge
    B = tmp / 'B1'
    run('merge', '--head-master', HM, '--cands', *S, '--scale', SCALE, '--head', H, '--out', B)
    bm = load_file(str(B / 'fp32-master' / 'fp32-master.safetensors'))
    want = {n: (hm[n].double() + SCALE * sum(real[k][n].reshape(NAMES[n]) for k in range(K))).float() for n in names}
    for n in names:
        assert bm[n].shape == hm[n].shape and torch.equal(bm[n], want[n]), n
    out = {}
    for f in ('model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'):
        out.update(load_file(str(B / f)))
    assert set(out) == set(names) | {FROZEN} and torch.equal(out[FROZEN], frozen)
    for n in names:
        assert out[n].dtype == torch.bfloat16 and torch.equal(out[n], want[n].to(torch.bfloat16)), n
    rep = json.loads((B / 'MERGE.json').read_text())
    assert rep['replaced'] == len(names) and rep['bf16_elements_changed'] > 0 and (B / 'config.json').exists()
    assert json.loads((B / 'model.safetensors.index.json').read_text())['weight_map'] == wm
    print('merge ok: relative update', round(rep['relative_update'], 4), 'bf16 elements changed', rep['bf16_elements_changed'])

    # --- chain (a): exactly the scaled single steps, applied one after another
    C, cur = [], {n: hm[n].clone() for n in names}
    for k in range(K):
        cur = {n: (cur[n].double() + SCALE * real[k][n].reshape(NAMES[n])).float() for n in names}
        p = tmp / f'C{k + 1}'; write_candidate(p, cur); C.append(p)
    run('chain', '--head-master', HM, '--singles', *S, '--chain', *C, '--scale', SCALE, '--out', tmp / 'chain_a.json')
    a = json.loads((tmp / 'chain_a.json').read_text())
    for s in a['steps']:
        assert s['cos'] > 0.999999 and s['rel_diff'] < 1e-4 and abs(s['norm_ratio'] - 1) < 1e-4, s
    w = a['whole_chain_vs_scaled_sum']; assert w['cos'] > 0.999999 and w['rel_diff'] < 1e-4, w
    for n in names:                                         # the end of this chain is the merged candidate (up to fp32 rounding)
        assert torch.allclose(cur[n], bm[n], rtol=0, atol=1e-5), n
    # --- chain (b): steps 2.. replaced by fresh noise of the same size -> step 1 still matches, later steps do not
    C2, cur = [], {n: hm[n].clone() for n in names}
    for k in range(K):
        step = {n: SCALE * real[k][n].reshape(NAMES[n]) if k == 0 else SCALE * 0.054 * torch.randn(NAMES[n]).double() for n in names}
        cur = {n: (cur[n].double() + step[n]).float() for n in names}
        p = tmp / f'D{k + 1}'; write_candidate(p, cur); C2.append(p)
    run('chain', '--head-master', HM, '--singles', *S, '--chain', *C2, '--scale', SCALE, '--out', tmp / 'chain_b.json')
    b = json.loads((tmp / 'chain_b.json').read_text())
    assert b['steps'][0]['cos'] > 0.999999 and b['steps'][0]['rel_diff'] < 1e-4
    assert all(abs(s['cos']) < 0.3 and s['rel_diff'] > 1 for s in b['steps'][1:]), b['steps']
    print('chain ok: exact chain rel_diff', '%.1e' % w['rel_diff'], '| perturbed chain step cos', [round(s['cos'], 3) for s in b['steps']])

    # --- error path: a candidate with a different tensor set is refused
    bad = tmp / 'bad'; write_candidate(bad, {n: hm[n] for n in names[:-1]})
    r = subprocess.run([sys.executable, str(HERE / 'merge_delta.py'), 'cos', '--head-master', str(HM), '--cands', str(S[0]), str(bad), '--out', str(tmp / 'x.json')],
                       capture_output=True, text=True)
    assert r.returncode != 0 and 'tensor names differ' in (r.stdout + r.stderr)
    print('ALL OK', tmp)


if __name__ == '__main__':
    main()
