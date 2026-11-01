"""Tests for clock_precheck.clock_verdict, plus (optional) a replay of every real reply to tasks 4713 / 5442.

  /tmp/mv090/bin/python -I test_clock_precheck.py
  /tmp/mv090/bin/python -I test_clock_precheck.py --tasks original-math7496.tasks.json --docs /data/k1/docs \
      --ff /data/k1/turns/{354,...,413}/batch.json
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clock_precheck import clock_verdict  # noqa: E402

G5442 = "8\\!:\\!34 \\text{ p.m.}"
P5442 = "The time right now is exactly midnight. What time will it be in 1234 minutes?"
G4713 = "4\\!:\\!51\\!:\\!06 \\text{ p.m.}"
P4713 = ('It is currently $3\\!:\\!00\\!:\\!00 \\text{ p.m.}$ What time will it be in $6666$ seconds?  (Enter the time in the '
         'format "HH:MM:SS", without including "am" or "pm".)')

# (gold, prompt, box, expected)
CASES = [
    (G5442, P5442, "\\boxed{8:34 \\text{ p.m.}}", True),
    (G5442, P5442, "\\boxed{8:34 \\text{ PM}}", True),
    (G5442, P5442, "\\boxed{8:34\\,\\text{pm}}", True),
    (G5442, P5442, "\\boxed{\\text{8:34 p.m.}}", True),
    (G5442, P5442, "\\boxed{20:34}", True),
    (G5442, P5442, "\\boxed{20{:}34}", True),
    (G5442, P5442, "\\boxed{20:34:00}", True),
    (G5442, P5442, "\\boxed{8:34 \\text{ a.m.}}", False),   # native accepts this today
    (G5442, P5442, "\\boxed{8:34}", False),                  # ambiguous; native accepts this today
    (G5442, P5442, "\\boxed{08:34}", False),                 # ambiguous; native accepts this today
    (G5442, P5442, "\\boxed{20:35}", False),
    (G5442, P5442, "\\boxed{9:34 \\text{ p.m.}}", False),
    (G5442, P5442, "\\boxed{1234}", False),
    (G5442, P5442, "\\boxed{20 \\text{ hours } 34 \\text{ minutes}}", False),
    (G4713, P4713, "\\boxed{4:51:06}", True),                # the prompt asks for no am/pm
    (G4713, P4713, "\\boxed{04:51:06}", True),
    (G4713, P4713, "\\boxed{16:51:06}", True),
    (G4713, P4713, "\\boxed{4:51:06 \\text{ p.m.}}", True),
    (G4713, P4713, "\\boxed{4:51:06 \\text{ a.m.}}", False),  # native accepts this today
    (G4713, P4713, "\\boxed{4:51:07}", False),
    (G4713, P4713, "\\boxed{4:51}", False),
    (G4713, P4713, "\\boxed{5:51:06}", False),
    (G4713, "", "\\boxed{4:51:06}", False),                   # without the prompt's instruction a bare 4:51:06 is ambiguous
    # gold is not a clock time: nothing changes
    ("\\frac{3}{4}", "", "\\boxed{3:4}", None),
    ("1:4", "", "\\boxed{1:4}", None),
    ("8", "", "\\boxed{8:34}", None),
    ("\\text{(D)}", "", "\\boxed{D}", None),
]


def unit():
    bad = 0
    for gold, prompt, box, want in CASES:
        got = clock_verdict(gold, box, prompt)
        flag = "" if got == want else "   <-- WRONG"
        bad += bool(flag)
        print(f"{gold[:22]:22s} {box:44s} want {str(want):5s} got {str(got):5s}{flag}")
    print(f"\n{len(CASES)} cases, {bad} wrong")
    return bad


def replay(a):
    """Every real reply to 4713 / 5442: the probe's grade today (native, then lenient v2) vs. with the clock check in front."""
    import lenient_grader as v1
    import lenient_grader_v2 as v2
    tasks = json.load(open(a.tasks))
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    SYS = tasks[0]["data"]["system_prompt"]
    byprob = {norm(t["data"]["problem"]): i for i, t in enumerate(tasks)}
    reps = []
    for doc in v1.iter_docs(a.docs):
        b = doc["batch"]
        if b["index"] in (4713, 5442):
            reps += [(b["index"], r["turns"][0]["text"]) for r in b["rollouts"] if r["turns"][0]["stop_reason"] == "eos"]
    for f in a.ff:
        for g in json.load(open(f))["groups"]:
            p = g["prompt"][len(SYS):] if g["prompt"].startswith(SYS) else g["prompt"]
            i = byprob.get(norm(p))
            if i in (4713, 5442):
                reps += [(i, r["output"]) for r in g["rollouts"] if r["reward"] != -1]
    for i, text in reps:
        d = tasks[i]["data"]
        box, _ = v1.live_project(text)
        if box is not None and v1.mv_equal(d["answer"], box):
            now = True
        else:
            gb, _, _ = v2.lenient_project(text)
            now = gb is not None and v2.lenient_equal(d["answer"], gb, d["prompt"])[0]
        b2 = box if box is not None else v2.lenient_project(text)[0]
        cv = clock_verdict(d["answer"], b2, d["prompt"]) if b2 is not None else None
        new = now if cv is None else cv
        print(f"task {i}: box={b2!r:40s} today={now} with-check={new}{'   <-- changes' if new != now else ''}")
    print(f"{len(reps)} real replies to 4713 / 5442")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks"); ap.add_argument("--docs", nargs="*", default=[]); ap.add_argument("--ff", nargs="*", default=[])
    a = ap.parse_args()
    bad = unit()
    if a.tasks:
        replay(a)
    sys.exit(1 if bad else 0)
