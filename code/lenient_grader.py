#!/usr/bin/env python3
"""affine_math verifier: reproduce the live grading, then measure a set of deterministic fixes.

Live grading = final-channel projection (subnet/mimo_answer_projection.py) + verify.py
(vendor/legacy/rollouts/envs/affine_math_v1/affine_math_v1/verify.py, math-verify 0.9.0:
the gold is parsed as \\boxed{gold}, the prediction is the last complete \\boxed{} of the reply).
Both are copied here at text level (the live projection counts <think>/</think> as tokens).

Fixes (each one is independent and only runs when the live grade is not "correct"):
  P1  projection: decide whether a `$` before the box opens or closes math (parity of unescaped `$`),
      so a bare \\boxed{} after a line that ends in closed `$...$` / `$$...$$` is no longer rejected
  P2  projection: text after the box, more than one box, a missing <think> or more than one </think>
      no longer zero the answer; the last complete box is graded as usual and a format violation is
      recorded (meant for a small separate penalty)
  E1  strip \\text{} / \\textbf{} / \\mathrm{} wrappers and thousands separators (`,\\!`, `{,}`)
  E2  compare letter lists as sets when the gold is a letter list
  E3  multiple choice: map a letter to the option text in the prompt, and drop an "(A)" label in front of a value
  E4  pmatrix column vector <-> tuple
  E5  single-variable inequality or `a \\neq c` <-> interval
  E6  k\\mathbf{I} / kI <-> k times the identity of the gold's size
  E7  complex numbers: treat `i` as the imaginary unit and check that the difference simplifies to 0

Needs the same pinned dependencies as verify.py:
  uv venv -p 3.12 /tmp/mv090
  VIRTUAL_ENV=/tmp/mv090 uv pip install math-verify==0.9.0 latex2sympy2-extended==1.11.0 \\
      sympy==1.14.0 antlr4-python3-runtime==4.13.2 mpmath==1.3.0
Usage:
  python fetch_k1.py --out /data/k1 --epochs 196-260
  /tmp/mv090/bin/python -I lenient_grader.py --tasks original-math7496.tasks.json \\
      --docs /data/k1/docs --out ../results/verifier/regression.json
"""
import argparse
import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

from math_verify import parse, verify

OPEN = "\\boxed{"


# ---------- live: final-channel projection (mimo_answer_projection.project / _boxed_final) ----------
def _box_end(text, start):
    depth = 1
    for at in range(start, len(text)):
        if text[at] == "{":
            depth += 1
        elif text[at] == "}":
            depth -= 1
            if depth == 0:
                return at
    return None


def live_project(text):
    """Return (grading_text, reason). grading_text is None when the live rule marks the reply unresolved."""
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return None, "think"
    a, b = text.index("<think>"), text.index("</think>")
    if a >= b or text[:a].strip():
        return None, "think"
    final = text[b + len("</think>"):].strip()
    if final.count(OPEN) != 1:
        return None, "n_boxed"
    begin = final.index(OPEN)
    at = _box_end(final, begin + len(OPEN))
    if at is None or not final[begin + len(OPEN):at].strip():
        return None, "unclosed"
    tail, prefix = final[at + 1:].strip(), final[:begin].rstrip()
    if tail:
        opening = {"\\]": "\\[", "\\)": "\\(", "$$": "$$", "$": "$"}.get(tail)
        if opening is None or not prefix.endswith(opening):
            return None, "tail"
    elif prefix.endswith(("\\[", "\\(", "$")):
        return None, "prefix"
    return final[begin:at + 1], "ok"


# ---------- fixes: projection ----------
def _unclosed_dollar(prefix):
    """True if the text before the box leaves a `$` / `$$` open (parity of unescaped `$`)."""
    return re.sub(r"\\\$", "", prefix).count("$") % 2 == 1


def lenient_project(text):
    """Return (grading_text, format_violation, fix). format_violation is None when the reply follows the protocol."""
    box, why = live_project(text)
    if box is not None:
        return box, None, None
    if why == "prefix":  # P1
        final = text.split("</think>")[-1].strip()
        begin = final.index(OPEN)
        if not _unclosed_dollar(final[:begin]) and not final[:begin].rstrip().endswith(("\\[", "\\(")):
            at = _box_end(final, begin + len(OPEN))
            return final[begin:at + 1], None, "P1"
    if why in ("tail", "n_boxed", "think", "prefix"):  # P2
        # no <think>: the whole reply is the answer channel; several </think>: use the last segment
        final = text.split("</think>")[-1]
        at0 = final.rfind(OPEN)
        if at0 >= 0:
            at = _box_end(final, at0 + len(OPEN))
            if at is not None and final[at0 + len(OPEN):at].strip():
                return final[at0:at + 1], why, "P2"
    return None, why, None


# ---------- live: verify.py ----------
_cache = {}


def mv_equal(gold, pred_box):
    """verify.py: True / False, or None where verify.py exits 75 (indeterminate)."""
    key = (gold, pred_box)
    if key not in _cache:
        try:
            g = parse(OPEN + gold + "}", raise_on_error=True)
            p = parse(pred_box, raise_on_error=True)
            _cache[key] = bool(g) and bool(verify(g, p, raise_on_error=True))
        except Exception:
            _cache[key] = None
    return _cache[key]


# ---------- fixes: answer equivalence ----------
def inner(box):
    return box[len(OPEN):-1]


def strip_text(s):
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox)\s*\{([^{}]*)\}", r"\1", s)
    for _ in range(2):  # 1,000,000 needs two passes
        s = re.sub(r"(\d),\\!(\d{3})", r"\1\2", s)
        s = re.sub(r"(\d)\{,\}(\d{3})", r"\1\2", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return s.strip()


LETTERS = re.compile(r"^\(?([A-Z])\)?(\s*,\s*\(?([A-Z])\)?)*$")


def letter_set(s):
    s = strip_text(s).replace("\\ ", " ").strip().strip("$")
    if LETTERS.match(s.replace(" ", "")):
        return frozenset(re.findall(r"[A-Z]", s))
    return None


def mc_options(prompt):
    """Options written as \\textbf{(A)}\\ xxx or \\text A. xxx -> {letter: text}."""
    opts = {}
    for m in re.finditer(r"\\textbf\{\(([A-E])\)\}\\?\s*(.*?)(?=\\qquad|\\textbf\{\([A-E]\)\}|\$\s*$|$)", prompt, re.S):
        opts[m.group(1)] = m.group(2).strip().rstrip("$").strip()
    if not opts:
        for m in re.finditer(r"\\text\s*([A-E])\.\s*(?:\\ )*\s*(.*?)(?=\\qquad|\\text\s*[A-E]\.|\$\$|$)", prompt, re.S):
            opts[m.group(1)] = m.group(2).strip()
    return opts


def pmatrix_to_tuple(s):
    def col(m):
        rows = [r.strip() for r in m.group(1).split("\\\\")]
        return "(" + ", ".join(rows) + ")" if all("&" not in r for r in rows) else m.group(0)
    return re.sub(r"\\begin\{[pb]matrix\}(.*?)\\end\{[pb]matrix\}", col, s, flags=re.S)


def ineq_to_interval(s):
    s = strip_text(s).replace(" ", "")
    m = re.fullmatch(r"[a-z](?:\\neq|\\ne)(-?[\w\\{}.]+)", s)
    if m:
        c = m.group(1)
        return f"(-\\infty,{c})\\cup({c},\\infty)"
    m = re.fullmatch(r"(-?[\w\\{}.]+)<[a-z]<(-?[\w\\{}.]+)", s)
    if m:
        return f"({m.group(1)},{m.group(2)})"
    m = re.fullmatch(r"[a-z](<|>|\\le|\\ge|\\leq|\\geq)(-?[\w\\{}.]+)", s)
    if m:
        op, c = m.groups()
        return {"<": f"(-\\infty,{c})", ">": f"({c},\\infty)", "\\le": f"(-\\infty,{c}]", "\\leq": f"(-\\infty,{c}]",
                "\\ge": f"[{c},\\infty)", "\\geq": f"[{c},\\infty)"}[op]
    return None


def scalar_identity(pred, gold):
    m = re.fullmatch(r"(-?\d+)\s*\\?(?:mathbf\{I\}|I)", strip_text(pred).replace(" ", ""))
    g = re.fullmatch(r"\\begin\{pmatrix\}(.*)\\end\{pmatrix\}", gold.strip(), re.S)
    if not (m and g):
        return None
    n, k = len(g.group(1).split("\\\\")), m.group(1)
    rows = [" & ".join(k if i == j else "0" for j in range(n)) for i in range(n)]
    return "\\begin{pmatrix} " + " \\\\ ".join(rows) + " \\end{pmatrix}"


def complex_equal(pred, gold):
    try:
        import sympy
        from latex2sympy2_extended import latex2sympy
        a = latex2sympy(strip_text(pred)).subs(sympy.Symbol("i"), sympy.I)
        b = latex2sympy(strip_text(gold)).subs(sympy.Symbol("i"), sympy.I)
        return sympy.simplify(sympy.expand(a - b)) == 0
    except Exception:
        return False


def lenient_equal(gold, pred_box, prompt):
    """Return (correct, fix). The live verify.py is tried first; fixes only run when it says no."""
    if mv_equal(gold, pred_box):
        return True, None
    pred = inner(pred_box)
    if mv_equal(strip_text(gold), OPEN + strip_text(pred) + "}"):
        return True, "E1"
    gs, ps = letter_set(gold), letter_set(pred)
    if gs and ps and gs == ps:
        return True, "E2"
    opts = mc_options(prompt)
    if opts:
        p = strip_text(pred)
        if re.match(r"^\s*\\?(?:text|textbf)?\{?\(?[A-E]\)", p):
            p = re.sub(r"^\s*\\?(?:text|textbf)?\{?\(?([A-E])\)?\}?\s*(\\\s)?\s*", "", p)
        ls = letter_set(pred)
        if ls and len(ls) == 1:
            p = opts.get(next(iter(ls)), p)
        if p and mv_equal(gold, OPEN + p + "}"):
            return True, "E3"
    if mv_equal(pmatrix_to_tuple(gold), OPEN + pmatrix_to_tuple(pred) + "}"):
        return True, "E4"
    iv = ineq_to_interval(pred)
    if iv and mv_equal(gold, OPEN + iv + "}"):
        return True, "E5"
    m = scalar_identity(pred, gold)
    if m and mv_equal(gold, OPEN + m + "}"):
        return True, "E6"
    if "i" in gold and "i" in pred and complex_equal(pred, gold):
        return True, "E7"
    return False, None


# ---------- run over all batch documents ----------
def iter_docs(dirs):
    seen = set()
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            raw = open(p, "rb").read()
            h = hashlib.sha256(raw).hexdigest()
            if h in seen:
                continue
            doc = json.loads(raw)
            if isinstance(doc, dict) and isinstance(doc.get("batch"), dict) and "rollouts" in doc["batch"]:
                seen.add(h)
                yield doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="original-math7496.tasks.json")
    ap.add_argument("--docs", required=True, nargs="+", help="directories with K=1 batch documents (searched recursively)")
    ap.add_argument("--out", required=True, help="output JSON")
    a = ap.parse_args()
    tasks = json.load(open(a.tasks))

    live_vs_repro = Counter()    # (live label, reproduced label)
    live_vs_fixed = Counter()    # (live label, label after fixes)
    by_fix = Counter()           # (live label, label after fixes, fixes used)
    violations = Counter()
    flips = defaultdict(Counter)
    n_docs = n_rollouts = 0
    epochs = set()
    for doc in iter_docs(a.docs):
        b = doc["batch"]
        n_docs += 1
        epochs.add(int(b["epoch"].rsplit("-e", 1)[-1]))
        data = tasks[b["index"]]["data"]
        gold, prompt = data["answer"], data["prompt"]
        for r in b["rollouts"]:
            n_rollouts += 1
            t = r["turns"][0]
            if t["stop_reason"] != "eos":
                continue  # capped replies are unresolved under every rule here
            live = r["classification"]
            box, _ = live_project(t["text"])
            ok = mv_equal(gold, box) if box is not None else None
            repro = "neutral" if ok is None else ("positive" if ok else "negative")
            live_vs_repro[(live, repro)] += 1

            gbox, viol, pfix = lenient_project(t["text"])
            if gbox is None:
                new, efix = "neutral", None
            else:
                ok2, efix = lenient_equal(gold, gbox, prompt)
                new = "positive" if ok2 else "negative"
            if viol:
                violations[viol] += 1
            live_vs_fixed[(live, new)] += 1
            if new != live:
                used = "+".join(x for x in (pfix, efix) if x) or "-"
                by_fix[(live, new, used)] += 1
                flips[(b["index"], live, new, used)][inner(gbox) if gbox else None] += 1

    out = {
        "scope": {"epochs": f"{min(epochs)}-{max(epochs)}", "n_epochs": len(epochs), "documents": n_docs,
                  "rollouts": n_rollouts, "completed_rollouts": sum(live_vs_repro.values())},
        "reproduction": {"live_vs_reproduced": {f"{k[0]}->{k[1]}": v for k, v in sorted(live_vs_repro.items())},
                         "mismatches": sum(v for (x, y), v in live_vs_repro.items() if x != y)},
        "after_fixes": {"live_vs_fixed": {f"{k[0]}->{k[1]}": v for k, v in sorted(live_vs_fixed.items())},
                        "changed_by_fix": {f"{k[0]}->{k[1]} via {k[2]}": v
                                           for k, v in sorted(by_fix.items(), key=lambda x: -x[1])},
                        "format_violations_kept_for_penalty": dict(violations)},
        "flips": [{"task": k[0], "gold": tasks[k[0]]["data"]["answer"], "live": k[1], "fixed": k[2], "fix": k[3],
                   "answers": dict(v.most_common())}
                  for k, v in sorted(flips.items(), key=lambda x: (x[0][3], -sum(x[1].values())))],
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("scope", "reproduction", "after_fixes")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
