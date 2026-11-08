"""v3 的新增规则（E10 / E11）：正例必须判对，反例（真的错答）必须仍判错。另外把 v2 的 50 例回归一遍。
用法: /tmp/mv090/bin/python -I test_lenient_grader_v3.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lenient_grader_v3 as v3

B = lambda s: "\\boxed{" + s + "}"
POS = [  # (gold, reply box, 来源)
    ("2007 + \\frac\\pi 2", "2007+\\frac{\\pi}{2}", "math:6807 unbraced frac in gold"),
    ("30^\\circ.", "30^\\circ", "math:2871 trailing period in gold"),
    ("30^\\circ.", "30", "math:2871"),
    ("6.4", "6.4\\%", "math:2376 percent"),
    ("[-1;4]", "[-1, 4]", "dapo:12899 semicolon interval"),
    ("x_{1}=1,x_{2}=-6", "x=-6 \\text{ or } x=1", "dapo:12701 assignment list"),
    ("\\begin{pmatrix} 1 & -1 \\\\ 1 & \\phantom -1 \\end{pmatrix}", "\\begin{pmatrix} 1 & -1 \\\\ 1 & 1 \\end{pmatrix}", "math:6782 phantom"),
    ("\\begin{pmatrix} 8/3 \\\\ 4/3 \\\\ -4/3 \\end{pmatrix}", "\\begin{pmatrix} \\dfrac{8}{3} \\\\[4pt] \\dfrac{4}{3} \\\\[4pt] -\\dfrac{4}{3} \\end{pmatrix}", "math:7362 row spacing"),
    ("\\begin{pmatrix} 1/4 \\\\ -7/4 \\\\ 1/2 \\end{pmatrix}", "\\begin{pmatrix} \\frac{1}{4} \\\\[4pt] -\\frac{7}{4} \\\\[4pt] \\frac{1}{2} \\end{pmatrix}", "math:7110"),
    ("\\begin{pmatrix} \\frac{1}{3} & -\\frac{2}{3} \\\\ -\\frac{2}{3} & \\frac{1}{3} \\end{pmatrix}", "\\begin{pmatrix}\\frac{1}{3}&-\\frac{2}{3}\\\\[4pt]-\\frac{2}{3}&\\frac{1}{3}\\end{pmatrix}", "math:7149-like"),
    ("\\frac12", "0.5", "unbraced digits"),
    ("x=1, x=3", "x = 3 \\text{ and } x = 1", "dapo:12369 roots in another order"),
    # sets and \pm: already equal under the live verify.py, kept here so a later change cannot break them
    ("x_{1}=1,x_{2}=-6", "\\{-6, 1\\}", "roots as a set (live)"),
    ("x=\\pm 2", "x=2 \\text{ or } x=-2", "\\pm gold, roots listed (live)"),
    ("x=2, x=-2", "x=\\pm 2", "roots listed gold, \\pm reply (live)"),
    ("\\pm 2", "2, -2", "bare \\pm gold (live)"),
]
NEG = [
    ("2007 + \\frac\\pi 2", "2007+\\frac{\\pi}{3}", "wrong frac"),
    ("30^\\circ.", "60^\\circ", "wrong angle"),
    ("6.4", "64\\%", "wrong percent"),
    ("6.4", "0.064", "decimal vs percent: stays wrong (gold asks percent)"),
    ("[-1;4]", "[-1, 5]", "wrong interval"),
    ("x_{1}=1,x_{2}=-6", "x=1", "missing root"),
    ("x_{1}=1,x_{2}=-6", "x=-6 \\text{ or } x=2", "wrong root"),
    ("x_{1}=1,x_{2}=-6", "x=1, x=1", "duplicate root"),
    ("A=5,B=10,C=15,D=30", "A=10,B=5,C=15,D=30", "different unknowns: values must stay attached"),
    ("x_{1}=1,x_{2}=-6", "\\{-6, 2\\}", "set with a wrong root"),
    ("x=\\pm 2", "x=2", "only one of the \\pm roots"),
    ("x=2, x=-2", "\\{2\\}", "set missing a root"),
    ("\\begin{pmatrix} 1 & -1 \\\\ 1 & \\phantom -1 \\end{pmatrix}", "\\begin{pmatrix} 1 & -1 \\\\ 1 & -1 \\end{pmatrix}", "phantom hides a minus, -1 is wrong"),
    ("\\begin{pmatrix} 8/3 \\\\ 4/3 \\\\ -4/3 \\end{pmatrix}", "\\begin{pmatrix} \\dfrac{8}{3} \\\\[4pt] \\dfrac{4}{3} \\\\[4pt] \\dfrac{4}{3} \\end{pmatrix}", "sign error"),
    ("9", "1, 9", "extra value (bad key, not a grader fix)"),
    ("50", "\\frac{50\\cdot 2^{100}}{2^{100}-4}", "close but not equal"),
    ("358926471", "358926714", "digit swap"),
]
# verify.py 自己就把 gold "[-1;4]" 判成等于 "(-1, 4)"（开闭区间不分）：lenient 层只在 live 判错时才跑，修不了，只能改答案键。
print("live verify.py: [-1;4] == (-1, 4)?", v3.mv_equal("[-1;4]", B("(-1, 4)")), " [-1,4] == (-1, 4)?", v3.mv_equal("[-1,4]", B("(-1, 4)")))
bad = 0
for g, p, why in POS:
    ok, fix = v3.lenient_equal(g, B(p), "")
    if not ok:
        bad += 1; print("POS FAIL", why, g, "|", p)
for g, p, why in NEG:
    ok, fix = v3.lenient_equal(g, B(p), "")
    if ok:
        bad += 1; print("NEG FAIL (accepted wrong)", why, g, "|", p, fix)
print(f"v3 new cases: {len(POS)} pos, {len(NEG)} neg, failures {bad}")

# v2 的 50 例（test_lenient_grader.py 的 CASES，带期望值）在 v3 下必须全对
import test_lenient_grader as t2
wrong = [name for name, gold, reply, prompt, want in t2.CASES if t2.grade(v3, gold, reply, prompt)[0] != want]
print("v2 cases:", len(t2.CASES), "v3 wrong on:", wrong)
sys.exit(1 if bad or wrong else 0)
