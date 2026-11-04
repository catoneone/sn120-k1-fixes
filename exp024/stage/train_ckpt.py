#!/usr/bin/env python3
"""kit/train_step.py 原样运行，只给可训练模型的每个 decoder 层加激活重算（torch.utils.checkpoint）。

原因：e600 文件（910 条、平均 5.3k token、46 条打满 12,288）在单张 H200 上用 kit 默认（padded、token_budget 16384）
第一个 micro-batch 就 OOM（139.8 GB）。重算只改执行，不改数学（层内无 dropout、模型在 eval 模式，前向重算得到相同的值）。
用法同 train_step.py。
"""
import os, sys
KIT = os.environ.get('KIT', '/data/affine/miner/kit'); sys.path.insert(0, KIT)
import torch, torch.utils.checkpoint as cp
import train_step

_orig = train_step.load_model


def load_model(path, device, dtype, trainable):
    m = _orig(path, device, dtype, trainable)
    if trainable:
        n = 0
        for mod in m.modules():
            if type(mod).__name__.endswith('DecoderLayer'):
                f = mod.forward
                def fwd(*a, _f=f, **k):
                    if torch.is_grad_enabled():
                        return cp.checkpoint(_f, *a, use_reentrant=False, **k)
                    return _f(*a, **k)
                mod.forward = fwd; n += 1
        print('activation checkpointing on %d decoder layers' % n, flush=True)
    return m


train_step.load_model = load_model
if __name__ == '__main__':
    train_step.main(sys.argv[1:])
