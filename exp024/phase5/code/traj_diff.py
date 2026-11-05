#!/usr/bin/env python3
"""EXP-024：线上 turn.json 的逐条训练记录 vs 本地 train_step 报告，按顺序对齐（核对 tokens / reward / advantage），看前向数值差在哪。
python3 traj_diff.py stage4/data/turn718.json results/phase4/step_M1.json"""
import json, sys, statistics as st
t = json.load(open(sys.argv[1]))['training']; L = json.load(open(sys.argv[2]))
lv, lc = t['trajectories'], L['trajectories']
assert len(lv) == len(lc)
bad = sum((a['tokens'], a['reward'], a['advantage'], a['nonce']) != (b['tokens'], b['reward'], b['advantage'], b['nonce']) for a, b in zip(lv, lc))
print('rollouts', len(lv), 'misaligned', bad)
def rows(key): return [(b[key] - a[key]) for a, b in zip(lv, lc) if a[key] is not None and b[key] is not None]
for key in ('old_logprob_mean', 'k3_kl_mean', 'support_logq_mean', 'loss', 'support_outside_tokens', 'loss_tokens', 'prompt_weight'):
    d = rows(key); rel = [abs(b[key] - a[key]) / max(1e-12, abs(a[key])) for a, b in zip(lv, lc) if isinstance(a[key], float) and a[key]]
    print('%-24s 本地−线上 均值 %+.3e  中位 %+.3e  |最大| %.3e  相对差中位 %s' % (key, st.mean(d), st.median(d), max(abs(x) for x in d), ('%.2e' % st.median(rel)) if rel else '-'))
print('derail', sum(a['derail_index'] is not None for a in lv), sum(b['derail_index'] is not None for b in lc))
# 按长度分档看 old_logprob_mean 的差
import math
bins = {}
for a, b in zip(lv, lc):
    k = min(4, int(math.log2(max(1, a['tokens'] / 1000)) + 1)) if a['tokens'] >= 1000 else 0
    bins.setdefault(k, []).append(b['old_logprob_mean'] - a['old_logprob_mean'])
for k in sorted(bins): print('  长度档', ['<1k', '1-2k', '2-4k', '4-8k', '>=8k'][k], 'n', len(bins[k]), 'Δlogp 均值 %+.2e 中位 %+.2e' % (st.mean(bins[k]), st.median(bins[k])))
print('KL 项：线上 k3 均值 %.4e 本地 %.4e' % (st.mean(a['k3_kl_mean'] for a in lv), st.mean(b['k3_kl_mean'] for b in lc)))
