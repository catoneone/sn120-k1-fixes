#!/usr/bin/env python3
"""EXP-024：线上候选和本地候选的 fp32 主权重更新（= 每个张量的梯度方向，normgrad 步）按层、按张量类型比 cos。
python3 layer_cos.py results/phase3/m_gate_vs_M1.json"""
import json, re, sys, collections, math
w = json.load(open(sys.argv[1]))['per_tensor']
grp = collections.defaultdict(lambda: [0.0, 0.0, 0.0])   # (layer 段, 类) -> dot, na², nb²
def cls(name):
    if 'linear_attn' in name:
        return 'GDN 门控(A_log/dt_bias/in_proj_a/b)' if re.search(r'A_log|dt_bias|in_proj_a|in_proj_b', name) else 'GDN 其他'
    if 'self_attn' in name: return '全注意力'
    if 'mlp' in name: return 'MLP'
    return '其他'
for t in w:
    m = re.match(r'model\.language_model\.layers\.(\d+)\.', t['name'])
    if not m or t['cos'] is None: continue
    L = int(m.group(1)); seg = '%02d-%02d' % (L // 8 * 8, L // 8 * 8 + 7)
    g = grp[(seg, cls(t['name']))]; g[0] += t['cos'] * t['na'] * t['nb']; g[1] += t['na'] ** 2; g[2] += t['nb'] ** 2
segs = sorted({k[0] for k in grp}); classes = ['GDN 门控(A_log/dt_bias/in_proj_a/b)', 'GDN 其他', '全注意力', 'MLP']
print('%-8s' % '层' + ''.join('%-38s' % c for c in classes))
for s in segs:
    print('%-8s' % s + ''.join('%-38s' % ('%.3f' % (grp[(s, c)][0] / math.sqrt(grp[(s, c)][1] * grp[(s, c)][2])) if grp[(s, c)][1] else '-') for c in classes))
