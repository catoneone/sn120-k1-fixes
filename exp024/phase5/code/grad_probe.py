#!/usr/bin/env python3
"""EXP-024 第五阶段 R3：同一个损失、同一批回答，不同前向 / 反向实现算出的梯度，逐张量存成 fp32。

  grad_probe.py --head DIR --batch batch718.json --hotkey SS58 --mode kit|live|live_fusednorm|fp32ref --n 24 --out grads_<mode>.safetensors

损失：L = Σ_i A_i · mean_t log p_t（全词表 log p，A_i = reward − 组均值），只对选中的 n 条回答（按长度三等分，每段 n/3 条，A ≠ 0，固定种子）。
模式：
  kit            kit 的 load_model（bf16，transformers 原生前向：fla chunk 核心 + torch F.conv1d + fla FusedRMSNormGated + sdpa）
  live           同上 + v83_fast_kernels.install（线上：fla causal_conv1d + torch 门控 RMSNorm 公式 + varlen FlashAttention）
  live_fusednorm 线上 kernel，但门控 RMSNorm 用 fla 融合 kernel（只换 norm，用来分离 norm 的影响）
  fp32ref        fp32 权重 + 纯 torch：torch_chunk_gated_delta_rule、F.conv1d、torch 门控 RMSNorm、sdpa（fp32），当作参考梯度
所有模式都开激活重算；每条回答单独反向，梯度在 fp32 累加。
"""
import argparse, json, os, random, sys, time
from pathlib import Path
import torch
KIT = os.environ.get('KIT', '/data/affine/miner/kit'); sys.path.insert(0, KIT)
import kitpath  # noqa: F401
import train_step, ff_format
import v83_fast_kernels as FK


def pick(groups, n, seed=0):
    rows = []
    for g in groups:
        rs = [float(r['reward']) for r in g['rollouts']]; m = sum(rs) / len(rs)
        for r in g['rollouts']:
            a = float(r['reward']) - m
            if a != 0: rows.append(dict(prompt=list(g['prompt_tokens']), out=list(r['tokens']), adv=a))
    rows.sort(key=lambda r: len(r['out'])); k = len(rows) // 3; rng = random.Random(seed); out = []
    for seg in (rows[:k], rows[k:2 * k], rows[2 * k:]):
        out += rng.sample(seg, n // 3)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--head', required=True); ap.add_argument('--batch', required=True)
    ap.add_argument('--hotkey', required=True); ap.add_argument('--mode', required=True); ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--out', required=True); a = ap.parse_args()
    tok = ff_format.tokenizer(a.head)
    v = ff_format.validate(Path(a.batch).read_bytes(), a.hotkey, tok, 125, 1000, v2=True)
    rows = pick(v['groups'], a.n)
    dev = torch.device('cuda'); fp32 = a.mode == 'fp32ref'
    model = train_step.load_model(a.head, dev, torch.float32 if fp32 else torch.bfloat16, True)
    if a.mode == 'live': FK.install(model)
    elif a.mode == 'live_fusednorm': FK.install(model, fused_norm=True)
    elif fp32:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
        for layer in model.model.language_model.layers:
            if layer.block_type == 'linear_attention':
                m = layer.linear_attn; m.chunk_gated_delta_rule = Q.torch_chunk_gated_delta_rule; m.causal_conv1d_fn = None
                m.norm.forward = (lambda h, g, _n=m.norm: FK.gated_rmsnorm_hf(_n, h, g))
    elif a.mode == 'live_gdn_only':          # 线上的 GDN kernel，全注意力保留 HF 原生（sdpa）
        for layer in model.model.language_model.layers:
            if layer.block_type == 'linear_attention':
                m = layer.linear_attn; m.forward = FK.make_gdn_forward(m, m.forward)
    elif a.mode == 'kit_flaconv':            # kit 路径，只把 torch F.conv1d 换成 fla causal_conv1d
        from fla.modules.convolution import causal_conv1d as FC
        def conv_fn(x, weight, bias=None, activation=None, seq_idx=None, **kw):
            y, _ = FC(x.transpose(1, 2), weight=weight, bias=bias, activation=activation)
            return y.transpose(1, 2)
        for layer in model.model.language_model.layers:
            if layer.block_type == 'linear_attention': layer.linear_attn.causal_conv1d_fn = conv_fn
    elif a.mode != 'kit':
        raise SystemExit('mode')
    train_step.enable_activation_checkpointing(model)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    acc = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params} if not fp32 else None
    from torch.utils.checkpoint import checkpoint
    t0 = time.time(); info = []
    for i, r in enumerate(rows):
        P, O = r['prompt'], r['out']
        ids = torch.tensor([P + O], device=dev)
        h = model.model(ids, use_cache=False).last_hidden_state[0, len(P) - 1:len(P) + len(O) - 1]
        tgt = torch.tensor(O, device=dev); lps = []
        def sel(hh, tt):
            lg = model.lm_head(hh).float(); return lg.gather(-1, tt[:, None]).squeeze(-1) - torch.logsumexp(lg, -1)
        for s in range(0, len(O), 2048):
            lps.append(checkpoint(sel, h[s:s + 2048], tgt[s:s + 2048], use_reentrant=False))
        lp = torch.cat(lps); loss = r['adv'] * lp.mean(); loss.backward()
        if acc is not None:
            for n, p in params:
                if p.grad is not None: acc[n] += p.grad.float(); p.grad = None
        info.append(dict(i=i, tokens=len(O), adv=r['adv'], logp_mean=float(lp.mean())))
        print(json.dumps(info[-1]), flush=True)
    G = acc if acc is not None else {n: p.grad.float() for n, p in params if p.grad is not None}
    from safetensors.torch import save_file
    save_file({n: t.detach().contiguous().cpu() for n, t in G.items()}, a.out)
    Path(a.out + '.json').write_text(json.dumps(dict(mode=a.mode, seconds=time.time() - t0, rows=info)))
    print('DONE', a.mode, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
