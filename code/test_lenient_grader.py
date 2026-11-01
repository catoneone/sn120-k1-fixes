"""Unit tests for the lenient affine_math grader: real correct answers that must pass, and deliberately wrong variants
that must not. Runs every case against v1 (the gate's allan_src.py = lenient_grader.py at 786cdb9) and v2
(lenient_grader_v2.py) and fails if v2 gets any case wrong.

  /tmp/mv090/bin/python -I test_lenient_grader.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lenient_grader as v1  # noqa: E402
import lenient_grader_v2 as v2  # noqa: E402

MC = ("$\\textbf{(A)}\\ 100 \\qquad \\textbf{(B)}\\ 110 \\qquad \\textbf{(C)}\\ 119 \\qquad "
      "\\textbf{(D)}\\ 120 \\qquad \\textbf{(E)}\\ 130$")
LINES = ("Which of the following describes the graph of $y=(f+g)(x)$?\n\n(A) a parabola tangent to the $x$-axis\n"
         "(B) a parabola not tangent to the $x$-axis\n(C) a horizontal line\n(D) a non-horizontal line\n"
         "(E) the graph of a cubic function")


def W(ans):
    return "<think>work</think>So the result follows.\n\n" + ans


# (name, gold, reply, prompt, expected correct?)
CASES = [
    # --- must stay correct (real answers from the K=1 and free-file batches)
    ("live: plain", "116", W("$\\boxed{116}$"), "", True),
    ("P1: $$ closed before a bare box", "-2^{49}", W("$$S = -2^{49}.$$\n\n\\boxed{-2^{49}}"), "", True),
    ("P2: period after the box", "819", W("The answer is $\\boxed{819}$."), "", True),
    ("P2: box inside a sentence", "3", W("the graph has $\\boxed{3}$ distinct $x$-intercepts."), "", True),
    ("P2: tail with a number", "6", W("there are $\\boxed{6}$ entries that are multiples of $67$."), "", True),
    ("P2: two </think>", "58", "<think>a</think>b</think>So $p+q=58$.\n\n$\\boxed{58}$", "", True),
    ("P2: no <think>", "58", "10! = 3628800 ... so $p+q=58$.\n\n$\\boxed{58}$", "", True),
    ("P2: same answer boxed twice", "73", W("Total $\\boxed{73}$.\n\n$\\boxed{73}$"), "", True),
    ("E1: \\text{} letter list", "\\text{B,E}", W("\\boxed{B, E}"), "", True),
    ("E1: thousands separator", "29,\\!322,\\!216", W("\\boxed{29322216}"), "", True),
    ("E3: letter -> option text", "119", W("\\boxed{C}"), MC, True),
    ("E3: label + value", "119", W("\\boxed{\\textbf{(C)}\\ 119}"), MC, True),
    ("E3: \\mathrm label + value", ".2", W("\\boxed{\\mathrm{(C) \\ }0.2}"), "", True),
    ("E3: label + value, no options", "\\frac{1}{7}\\triangle ABC", W("\\boxed{\\text{(C) } \\frac{1}{7}\\triangle ABC}"), "", True),
    ("E4: tuple for a column vector", "\\begin{pmatrix} 6 \\\\ -17 \\end{pmatrix}", W("\\boxed{(6, -17)}"), "", True),
    ("E5: \\neq", "(-\\infty,3) \\cup (3,\\infty)", W("\\boxed{a \\neq 3}"), "", True),
    ("E5: \\leq", "(-\\infty,3]", W("\\boxed{x \\leq 3}"), "", True),
    ("E5: \\geq negative", "[-2,\\infty)", W("\\boxed{x\\geq -2}"), "", True),
    ("E6: kI", "\\begin{pmatrix} -5 & 0 \\\\ 0 & -5 \\end{pmatrix}", W("\\boxed{-5\\mathbf{I}}"), "", True),
    ("E7: complex order", "-32 + 32i \\sqrt{3}", W("\\boxed{-32 + 32\\sqrt{3}i}"), "", True),
    ("E8: gold letter, reply letter + text", "\\text{(D)}", W("\\boxed{\\text{(D) a non-horizontal line}}"), LINES, True),
    ("E8: gold letter, reply letter + formula", "\\text{(C)}", W("\\boxed{\\text{(C) } p + s = 0}"), "", True),
    ("E9: 24h clock", "8\\!:\\!34 \\text{ p.m.}", W("\\boxed{20:34}"), "", True),
    ("E9: 24h clock with seconds", "4\\!:\\!51\\!:\\!06 \\text{ p.m.}", W("\\boxed{16:51:06}"), "", True),
    ("E9: same clock with p.m.", "8\\!:\\!34 \\text{ p.m.}", W("\\boxed{8:34 \\text{ p.m.}}"), "", True),
    # --- must stay wrong
    ("hedge: last box is the gold", "5", W("$\\boxed{3}$ or maybe $\\boxed{5}$"), "", False),
    ("hedge: first box is the gold", "5", W("$\\boxed{5}$ or maybe $\\boxed{3}$"), "", False),
    ("no <think>, boxes disagree", "5", "Try \\boxed{3}? No. Final: \\boxed{5}", "", False),
    ("tail retracts the answer", "5", W("$\\boxed{5}$. Wait, that is wrong; the answer is 7."), "", False),
    ("tail gives another answer", "5", W("$\\boxed{5}$, but actually it is 7."), "", False),
    ("E3: wrong letter", "119", W("\\boxed{B}"), MC, False),
    ("E3: wrong label, right value", "119", W("\\boxed{\\text{(B)}\\ 119}"), MC, False),
    ("E3: right label, wrong value", "119", W("\\boxed{\\text{(C)}\\ 120}"), MC, False),
    ("E2: subset of letters", "\\text{B, E}", W("\\boxed{B}"), "", False),
    ("E2: superset of letters", "\\text{B, E}", W("\\boxed{B, D, E}"), "", False),
    ("E1: thousands, last digit off", "29,\\!322,\\!216", W("\\boxed{29,322,217}"), "", False),
    ("E4: tuple reversed", "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}", W("\\boxed{(2, 1)}"), "", False),
    ("E5: open vs closed end", "(-\\infty,3]", W("\\boxed{x<3}"), "", False),
    ("E5: 2<x<4 vs [2,4]", "[2,4]", W("\\boxed{2<x<4}"), "", False),
    ("E5: wrong \\neq", "(-\\infty,3)\\cup(3,\\infty)", W("\\boxed{a\\neq 2}"), "", False),
    ("E5: \\geq vs \\leq", "[-2,\\infty)", W("\\boxed{x\\leq -2}"), "", False),
    ("E6: wrong k", "\\begin{pmatrix} -5 & 0 \\\\ 0 & -5 \\end{pmatrix}", W("\\boxed{5\\mathbf{I}}"), "", False),
    ("E7: conjugate", "-32 + 32i \\sqrt{3}", W("\\boxed{-32 - 32\\sqrt{3}i}"), "", False),
    ("E7: pi expression", "\\frac{\\pi}{2}", W("\\boxed{\\frac{\\pi}{3}}"), "", False),
    ("E7: interval with infinity", "(-\\infty, 2)", W("\\boxed{(-\\infty, 3)}"), "", False),
    ("E8: wrong letter + text", "\\text{(D)}", W("\\boxed{\\text{(C) a horizontal line}}"), LINES, False),
    ("E8: right letter, other option's text", "\\text{(D)}", W("\\boxed{\\text{(D) a horizontal line}}"), LINES, False),
    ("E9: off by a second", "4\\!:\\!51\\!:\\!06 \\text{ p.m.}", W("\\boxed{16:51:07}"), "", False),
    ("E9: off by a minute", "8\\!:\\!34 \\text{ p.m.}", W("\\boxed{20:35}"), "", False),
    ("Wed vs Wednesday stays wrong", "\\text{Wednesday}", W("\\boxed{\\text{Wed}}"), "", False),
]


# Wrong answers the LIVE verify.py (math-verify 0.9.0) already accepts, so no lenient layer that only runs on a non-positive
# live grade can catch them. Reported, not asserted.
LIVE_FALSE_POSITIVES = [
    ("clock: 8:34 for 8:34 p.m.", "8\\!:\\!34 \\text{ p.m.}", W("\\boxed{8:34}"), ""),
    ("clock: 8:34 a.m. for 8:34 p.m.", "8\\!:\\!34 \\text{ p.m.}", W("\\boxed{8:34 \\text{ a.m.}}"), ""),
]


def grade(mod, gold, reply, prompt):
    box, _, pfix = mod.lenient_project(reply)
    if box is None:
        return False, "unresolved"
    ok, efix = mod.lenient_equal(gold, box, prompt)
    return ok, "+".join(x for x in (pfix, efix) if x) or "live"


def main():
    bad_v1, bad_v2 = [], []
    print(f"{'case':42s} {'want':5s} {'v1':14s} {'v2':14s}")
    for name, gold, reply, prompt, want in CASES:
        r1, r2 = grade(v1, gold, reply, prompt), grade(v2, gold, reply, prompt)
        m1 = "" if r1[0] == want else "  <-- v1 wrong"
        m2 = "" if r2[0] == want else "  <-- v2 WRONG"
        print(f"{name:42s} {str(want):5s} {str(r1[0]) + ' ' + r1[1]:14s} {str(r2[0]) + ' ' + r2[1]:14s}{m1}{m2}")
        (bad_v1.append(name) if m1 else None), (bad_v2.append(name) if m2 else None)
    for name, gold, reply, prompt in LIVE_FALSE_POSITIVES:
        box, _ = v1.live_project(reply)
        print(f"live verify.py false positive? {name:34s} -> {bool(box and v1.mv_equal(gold, box))}")
    print(f"\n{len(CASES)} cases; v1 wrong on {len(bad_v1)}: {bad_v1}\nv2 wrong on {len(bad_v2)}: {bad_v2}")
    sys.exit(1 if bad_v2 else 0)


if __name__ == "__main__":
    main()
