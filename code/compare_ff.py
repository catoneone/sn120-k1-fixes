#!/usr/bin/env python3
"""Summary for run_ff_steps.sh: every arm against the start, and the end of each chain against the one-step candidates of the
same file (paired per task: all arms share the gate seeds, so candidate-vs-candidate is a common-random-numbers comparison).
Gate tasks whose problem text appears in the file are left out (gate_exclude.json, written by each arm).
Usage: python compare_ff.py OUT
"""
import glob, json, math, os, re, sys
from collections import defaultdict

OUT = sys.argv[1]


def load(path):
    return [json.loads(x) for x in open(path) if x.strip()]


def per_task(rows, excl):
    acc = defaultdict(list)
    for r in rows:
        if r['index'] not in excl:
            acc[r['index']].append(float(r['correct']))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def paired(a_rows, b_rows, excl):
    """b minus a: mean per-task difference (pp), z, and b's capped rate."""
    a, b = per_task(a_rows, excl), per_task(b_rows, excl)
    d = [b[k] - a[k] for k in a if k in b]
    n = len(d); m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else 0.0
    cap = sum(r['capped'] for r in b_rows if r['index'] not in excl) / max(1, sum(r['index'] not in excl for r in b_rows))
    return 100 * m, (m / se if se else 0.0), 100 * cap, n


start = load(os.path.join(OUT, 'start', 'gate.jsonl'))
arms = defaultdict(dict)            # file tag -> arm -> {step: rows}
excl = defaultdict(set)
for d in sorted(glob.glob(os.path.join(OUT, 'F*_*'))):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d); tag, arm = name.split('_', 1)
    ex = os.path.join(d, 'gate_exclude.json')
    if os.path.exists(ex):
        excl[tag] |= set(json.load(open(ex)))
    for f in glob.glob(os.path.join(d, 'gate_s[0-9][0-9].jsonl')):
        s = int(re.search(r'gate_s(\d+)\.jsonl$', f).group(1))
        if s:
            arms[tag].setdefault(arm, {})[s] = load(f)

for tag in sorted(arms):
    ex = excl[tag]
    print(f'== {tag}  (gate tasks excluded for overlap with the file: {len(ex)})')
    ends = {}
    for arm in sorted(arms[tag]):
        cells = []
        for s in sorted(arms[tag][arm]):
            dpp, z, cap, n = paired(start, arms[tag][arm][s], ex)
            cells.append(f's{s} {dpp:+.1f} (z {z:+.1f}, capped {cap:.0f}%)')
        ends[arm] = arms[tag][arm][max(arms[tag][arm])]
        print(f'  {arm:9s} vs start: ' + '   '.join(cells))
    for a, b in (('P_ours', 'C_ours'), ('P_live', 'C_ours'), ('P_live', 'C_live'), ('C_live', 'C_ours'), ('P_live', 'P_ours'),
                 ('C_ours', 'C3_ours')):
        if a in ends and b in ends:
            dpp, z, cap, n = paired(ends[a], ends[b], ex)
            print(f'  end {b} minus end {a}: {dpp:+.1f}pp (z {z:+.1f}, n {n})')
