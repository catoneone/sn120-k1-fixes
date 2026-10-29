#!/usr/bin/env python3
"""EXP-023 汇总：从 HF 私有仓库 alllwang/sn120-exp023 拉回 results/，打印单步和累积两张表。
用法：.venv/bin/python -I experiments/EXP-023-bf16-rounding/summarize.py [--no-pull]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
F = ('ll_correct', 'll_wrong', 'll_capped', 'logPS', 'logPS_capped', 'eos_think')


def pull():
    from huggingface_hub import snapshot_download
    snapshot_download('alllwang/sn120-exp023', repo_type='dataset', local_dir=str(RES))


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--no-pull', action='store_true')
    a = ap.parse_args()
    if not a.no_pull:
        pull()
    print('## 单步（同一个官方梯度，三种处理）\n')
    for f in sorted((RES / 'stage1').glob('stage1_e*.json')):
        r = json.loads(f.read_text())
        g = r['grad']
        print(f"### e{r['turn']}（{r['head'][:8]} → {r['candidate'][:8]}）")
        print(f"- 复算：objective {g['objective']:.6g}（官方 {g['objective_published']:.6g}），裁剪前范数 {g['preclip']:.5f}（官方 {g['preclip_published']:.5f}），"
              f"rtn 改动 {r['rtn']['changed']:,}（官方 {r['changed_elements_published']:,}），与官方 candidate 不同的元素 {r['candidate_mismatched_elements']:,}；"
              f"rtn 改动里 |w|<2^-12 的 {r['rtn_changed_small_weights'] / max(r['rtn']['changed'], 1):.1%}")
        print('\n| | fp32（原本的一步） | rtn（官方） | sr |\n|---|---|---|---|')
        for k, lab in (('changed_frac', '改动元素占比'), ('norm_ratio', '留下的范数 / 原本'), ('cos_intended', '与原本方向的 cos'),
                       ('first_order_loss_change', '一阶 loss 变化（这一轮批次）')):
            print(f"| {lab} | " + ' | '.join(f"{r[m][k]:.4g}" for m in ('fp32', 'rtn', 'sr')) + ' |')
        lin = r['probe_linear_delta']
        for k in F:
            print(f"| 一阶 Δ{k}（固定文本） | " + ' | '.join(f"{lin[m][k]:.3g}" for m in ('fp32', 'rtn', 'sr')) + ' |')
        print(f"| rtn 的真实前向 Δ（对照一阶） | | " + ', '.join(f"{k} {r['probe_direct_delta_rtn'][k]:.2g}" for k in F) + ' | |\n')
    print('## 17 步累积（固定文本上的 F，及发布的 bf16 权重相对起点、和崩溃方向的符号一致率）\n')
    print('| arm | step | ' + ' | '.join(F) + ' | 一致率 b1 | 一致率 orig | 主权重一致率 b1 |\n|' + '---|' * (len(F) + 5))
    for arm in ('rtn', 'fp32', 'fp32_renorm', 'sr', 'randsign', 'fp32_renorm_normgrad', 'fp32_renorm_pos'):
        for s in jl(RES / f'stream_{arm}' / 'snapshots.jsonl'):
            al = s['align']
            print(f"| {arm} | {s['step']} | " + ' | '.join(f"{s['probe'][k]:.5g}" for k in F) +
                  f" | {al['collapse_b1']['published_agree']:.3f} | {al['collapse_orig']['published_agree']:.3f} | "
                  f"{al['collapse_b1']['master_agree'] if al['collapse_b1']['master_agree'] is None else round(al['collapse_b1']['master_agree'], 3)} |")
    print('\n每步：')
    for arm in ('rtn', 'fp32', 'fp32_renorm', 'sr', 'randsign', 'fp32_renorm_normgrad', 'fp32_renorm_pos'):
        st = jl(RES / f'stream_{arm}' / 'steps.jsonl')
        if st:
            ch = [x['changed'] for x in st]; ri = [x['realized_over_intended'] for x in st]
            gi = sum(x['first_order_loss_change_intended'] for x in st); gr = sum(x['first_order_loss_change_realized'] for x in st)
            out = sum(x.get('support_outside_tokens') or 0 for x in st); tok = sum(x.get('loss_tokens') or 0 for x in st)
            print(f"- {arm}: {len(st)} 步；每步改动 {min(ch):,}–{max(ch):,}；发布变化 / 原本 {min(ri):.3f}–{max(ri):.3f}；"
                  f"一阶 loss 变化合计 原本 {gi:.4g} / 实际发布 {gr:.4g}；候选集外 token {out}/{tok}")
    print('\n## held-out 生成（750 题 × 2）\n')
    print('| ckpt | 截断：通过率 | 截断：打满 | 全词表：通过率 | 全词表：打满 | 全词表：EOS 无答案 |\n|---|---|---|---|---|---|')
    for d in sorted(RES.glob('gen_*')):
        sfile = d / 'summary.json'
        if sfile.exists():
            s = json.loads(sfile.read_text()); t, f = s.get('trunc', {}), s.get('full', {})
            print(f"| {d.name[4:]} | {t.get('pass_rate', float('nan')):.3f} | {t.get('capped', float('nan')):.3f} | "
                  f"{f.get('pass_rate', float('nan')):.3f} | {f.get('capped', float('nan')):.3f} | {f.get('eos_no_answer', float('nan')):.3f} |")


if __name__ == '__main__':
    main()
