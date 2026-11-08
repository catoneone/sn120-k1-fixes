"""v2 vs v3 on real replies: every public GREEN batch file (e354-e903), graded the way the probe grades.
Native grade first (live projection + verify.py, as in allan_src.py); the lenient layer runs only when that is not
positive. Prints the count of every change and writes each changed reply to --out.

  /tmp/mv090/bin/python -I regress_v3.py --batches DIR_WITH_<epoch>.json --pool gate-pool-v2.jsonl \
      --allan-src miner/validator/probe/allan_src.py --out ../results/verifier/regression_v3.json

Expected on the 64 files e354-e903 (30,470 replies): 18 changed, all wrong -> correct
(E10 matrix row spacing 16, E10 [-1;4] 1, E11 1), 0 correct -> wrong.
Needs math-verify 0.9.0, latex2sympy2_extended 1.11.0, sympy 1.14.0 (the gate pins).
"""
import argparse, collections, glob, json, os, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ARGS = None
M = {}


def load_pool(path):
    pool, by_head = {}, collections.defaultdict(list)
    for line in open(path):
        t = json.loads(line); pool[t['id']] = t
        by_head[t['problem'].strip()[:80]].append(t['id'])
    return pool, by_head


def match(prompt, pool, by_head):
    """Batch prompts are MANDATE + blank line + the pool problem text."""
    body = prompt.split('\n\n', 1)[1].strip() if '\n\n' in prompt else prompt.strip()
    ids = by_head.get(body[:80], [])
    return next((i for i in ids if pool[i]['problem'].strip() == body), ids[0] if ids else None)


def init(allan_src):
    ns = {'__name__': 'allan_lenient_v1'}
    exec(compile('import re\nfrom math_verify import parse, verify\n' + open(allan_src).read(), 'allan_lenient_v1', 'exec'), ns)
    import lenient_grader_v2 as v2, lenient_grader_v3 as v3
    M.update(v1=ns, v2=v2, v3=v3)


def one(job):
    gold, prompt, text = job
    v1 = M['v1']
    live_box, _ = v1['live_project'](text)
    if live_box is not None and v1['mv_equal'](gold, live_box):
        return 'live', 'live'
    res = []
    for mod in (M['v2'], M['v3']):
        box, _, pfix = mod.lenient_project(text)
        ok, efix = mod.lenient_equal(gold, box, prompt) if box is not None else (False, None)
        res.append(('+'.join(x for x in (pfix, efix) if x) or 'live') if ok else ('negative' if live_box is not None else 'unresolved'))
    return tuple(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', required=True)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--allan-src', required=True)
    ap.add_argument('--out', default='regression_v3.json')
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    pool, by_head = load_pool(a.pool)
    jobs, meta, files = [], [], sorted(glob.glob(os.path.join(a.batches, '*.json')), key=lambda f: int(os.path.basename(f)[:-5]))
    for f in files:
        for g in json.load(open(f)).get('groups', []):
            tid = match(g['prompt'], pool, by_head)
            if tid:
                for r in g['rollouts']:
                    jobs.append((pool[tid]['answer'], pool[tid]['problem'], r['output'])); meta.append((tid, os.path.basename(f)[:-5], r['output']))
    with Pool(a.procs, initializer=init, initargs=(a.allan_src,)) as p:
        res = p.map(one, jobs, chunksize=50)
    pos = lambda x: x not in ('negative', 'unresolved')
    changes, C = [], collections.Counter()
    for (tid, ep, text), (r2, r3) in zip(meta, res):
        if pos(r2) != pos(r3):
            changes.append({'task': tid, 'epoch': ep, 'gold': pool[tid]['answer'], 'v2': r2, 'v3': r3,
                            'final_tail': text.split('</think>')[-1][-300:]})
            C[('wrong -> correct', r3) if pos(r3) else ('correct -> wrong', r2)] += 1
    json.dump(changes, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f'files {len(files)}  replies {len(jobs)}  changed {len(changes)}')
    for k, v in sorted(C.items()):
        print(f'  {k[0]:17s} {k[1]:10s} {v}')


if __name__ == '__main__':
    main()
