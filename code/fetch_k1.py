#!/usr/bin/env python3
"""下载 K=1 各轮的 turn.json 和被训练的矿工批次原文（按 sha256 命名，供 replay 的 --docs-dir 用）。
affine.io 拦 Python urllib 的默认 UA，所以用 curl。
用法：python fetch_k1.py --out /data/k1 --epochs 196-213
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BASE = 'https://affine.io'


def curl(url, dst):
    subprocess.run(['curl', '-sSL', '--fail', '--retry', '3', '--max-time', '180', '-A', 'sn120-replay/1', '-o', str(dst), url],
                   check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--epochs', required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.epochs.split('-'))
    out = Path(a.out); (out / 'turns').mkdir(parents=True, exist_ok=True); (out / 'docs').mkdir(parents=True, exist_ok=True)
    for e in range(lo, hi + 1):
        tp = out / 'turns' / f'{e}.json'
        if not tp.exists():
            curl(f'{BASE}/k1/turns/{e}/turn.json', tp)
        turn = json.loads(tp.read_text())
        n = 0
        for row in turn['dispatch']['selected']:
            d = row['document']
            dp = out / 'docs' / (d['sha256'] + '.json')
            if not dp.exists() or hashlib.sha256(dp.read_bytes()).hexdigest() != d['sha256']:
                url = d['url'] if d['url'].startswith('http') else BASE + d['url']
                curl(url, dp)
            if hashlib.sha256(dp.read_bytes()).hexdigest() != d['sha256']:
                raise SystemExit(f'e{e}: document hash mismatch {d["sha256"]}')
            n += 1
        print('K1', e, 'docs', n, 'head', turn['training']['input']['checkpoint'][:8], 'candidate', turn['candidate']['id'][:8],
              'promoted', turn['promoted'], flush=True)


if __name__ == '__main__':
    main()
