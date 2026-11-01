"""v1 (gate's allan_src = lenient_grader.py at 786cdb9) vs v2 on real replies: the K=1 batch documents (e196-e260, the
v1 regression set) and the free-file GREEN batch files (e381-e413). Prints every reply whose grade changes.

  /tmp/mv090/bin/python -I regress_v2.py --tasks original-math7496.tasks.json --docs /data/k1/docs \
      --ff /data/k1/turns/381/batch.json ... --out ../results/verifier/regression_v2.json
"""
import argparse, json, re, sys, os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lenient_grader as v1
import lenient_grader_v2 as v2


def grade(mod, live_box, text, gold, prompt):
    """The probe: native grade first, the lenient layer only when it is not positive."""
    if live_box is not None and v1.mv_equal(gold, live_box):
        return "positive", "live"
    box, _, pfix = mod.lenient_project(text)
    if box is not None:
        ok, efix = mod.lenient_equal(gold, box, prompt)
        if ok:
            return "positive", "+".join(x for x in (pfix, efix) if x)
    return ("negative" if live_box is not None else "neutral"), "live"


def replies(a, tasks):
    for doc in v1.iter_docs(a.docs):
        b = doc["batch"]
        for r in b["rollouts"]:
            t = r["turns"][0]
            if t["stop_reason"] == "eos":
                yield "k1", b["index"], t["text"]
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    SYS = tasks[0]["data"]["system_prompt"]
    byprob = {norm(t["data"]["problem"]): i for i, t in enumerate(tasks)}
    for f in a.ff:
        for g in json.load(open(f))["groups"]:
            p = g["prompt"][len(SYS):] if g["prompt"].startswith(SYS) else g["prompt"]
            for r in g["rollouts"]:
                if r["reward"] == -1:  # the miner's "did not finish" label: capped, never reaches the lenient layer
                    continue
                yield "ff", byprob[norm(p)], r["output"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True); ap.add_argument("--docs", nargs="+", default=[])
    ap.add_argument("--ff", nargs="*", default=[]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = json.load(open(a.tasks))
    n = Counter(); trans = Counter(); changes = []
    for src, i, text in replies(a, tasks):
        d = tasks[i]["data"]; live_box, _ = v1.live_project(text)
        g1, h1 = grade(v1, live_box, text, d["answer"], d["prompt"])
        g2, h2 = grade(v2, live_box, text, d["answer"], d["prompt"])
        n[src] += 1; trans[(src, g1, g2)] += 1
        if g1 != g2:
            box = v2.lenient_project(text)[0] or v1.lenient_project(text)[0]
            changes.append(dict(src=src, task=i, gold=d["answer"], v1=f"{g1} {h1}", v2=f"{g2} {h2}",
                                box=box, tail=text.split("</think>")[-1][-400:]))
    out = dict(replies=dict(n), transitions={f"{s}: {x}->{y}": c for (s, x, y), c in sorted(trans.items())}, changes=changes)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out["replies"]), json.dumps(out["transitions"], indent=1))
    for c in changes:
        print(f"[{c['src']}] task {c['task']} gold={c['gold']!r} | {c['v1']} -> {c['v2']} | box={c['box']!r}")


if __name__ == "__main__":
    main()
