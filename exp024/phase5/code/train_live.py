#!/usr/bin/env python3
"""kit/train_step.py 原样运行，但换成线上训练器的 kernel 和批处理：actor 与 KL 参考模型都装上 v83_fast_kernels.install
（fla causal_conv1d + chunk_gated_delta_rule varlen、torch 门控 RMSNorm 公式、varlen FlashAttention），
batched_log_probs 在 engine='packed' 时改用 fast_kernels.packed_log_probs，train_update_batched 强制 engine='packed'。
也就是线上 v83_dp 每个分片做的事，只是 1 张卡、分片之间不再 NCCL 求和（求和顺序不同）。用法同 train_step.py。
"""
import functools, json, os, sys
KIT = os.environ.get('KIT', '/data/affine/miner/kit'); sys.path.insert(0, KIT)
import kitpath  # noqa: F401
import train_step
import v8211_batched as B
import v83_fast_kernels as FK

_load = train_step.load_model


def load_model(path, device, dtype, trainable):
    m = _load(path, device, dtype, trainable)
    print(json.dumps(dict(fast_kernels=FK.install(m), trainable=trainable)), flush=True)
    return m


_blp = B.batched_log_probs


def batched_log_probs(runtime, items, *, gradients, engine, logit_chunk=2048, support=None):
    if engine == 'packed':
        return FK.packed_log_probs(runtime, items, gradients=gradients, logit_chunk=logit_chunk, support=support)
    return _blp(runtime, items, gradients=gradients, engine=engine, logit_chunk=logit_chunk, support=support)


_tub = B.train_update_batched


@functools.wraps(_tub)
def train_update_batched(*a, **k):
    k['engine'] = 'packed'
    return _tub(*a, **k)


train_step.load_model = load_model
B.batched_log_probs = batched_log_probs
B.train_update_batched = train_update_batched
if __name__ == '__main__':
    train_step.main(sys.argv[1:])
