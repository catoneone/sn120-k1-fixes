"""allan-lenient-v2: drop-in replacement for the gate's allan_src.py (allan-lenient-v1-786cdb9).

Same entry points and signatures: lenient_project(text) -> (box, violation, fix), lenient_equal(gold, box, prompt) -> (ok, fix).
Changes against v1 (each one found by test_lenient_grader.py or by the free-file GREEN batches e381-e413):
  P2  more than one different box in the answer channel: correct only if every box is correct (v1 graded the last box, so a
      hedge "\\boxed{3} or \\boxed{5}" scored as 5); text after the box: not graded if it takes the answer back
      ("Wait, ...", "actually", "but", "wrong", "instead", another \\boxed)
  E3  an "(X)" label in front of a value: the value has to match the gold AND, when the options parse, option X has to be the gold
      (v1 accepted "(B) 119" when 119 is option C); options written one per line as "(A) text" now parse too;
      \\mathrm{(C) \\ }0.2 style labels are recognised
  E5  `x \\leq 3` / `x \\geq -2`: the operator alternation matched `\\le` first and left `q3`; longest operator first now
  E8  the gold is an option letter and the reply is the letter plus the option text: "(D) a non-horizontal line" for (D)
  E9  clock times: 20:34 = 8:34 p.m., 16:51:06 = 4:51:06 p.m. (only when both sides are unambiguous: a.m./p.m. given, or hour 0 or >= 13)
"""
import re
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


RETRACT = re.compile(r"\\boxed|\b(?:wait|actually|however|but|wrong|incorrect|mistake|instead|correction|re-?check)\b", re.I)


def _boxes(text):
    out, at = [], text.find(OPEN)
    while at >= 0:
        end = _box_end(text, at + len(OPEN))
        if end is None:
            break
        out.append(text[at:end + 1])
        at = text.find(OPEN, end + 1)
    return out


class Box(str):
    """The graded (last) box; .others = the other, different boxes of the answer channel. Every one of them has to be correct
    too (lenient_equal checks), so a hedge "\\boxed{3} or \\boxed{5}" fails while "\\boxed{3\\sec\\frac12\\theta} ... \\boxed{D}" passes."""
    others = ()


def _with_others(final, box):
    norm = lambda b: re.sub(r"\s+", "", inner(b))
    seen, others = {norm(box)}, []
    for b in _boxes(final):
        if norm(b) not in seen:
            seen.add(norm(b)); others.append(b)
    out = Box(box); out.others = tuple(others)
    return out


def _retracts(tail):
    """Text after the box that takes the answer back or gives another one."""
    return bool(RETRACT.search(tail))


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
                if _retracts(final[at + 1:]):
                    return None, why, None
                return _with_others(final, final[at0:at + 1]), why, "P2"
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
    if not opts:
        for m in re.finditer(r"^\s*\(([A-E])\)\s*(.+?)\s*$", prompt, re.M):
            opts[m.group(1)] = m.group(2).strip()
    return opts


LABEL = re.compile(r"^\s*\(?([A-E])\)\s*(?:\\ |~|\s)*(.*)$", re.S)


def option_label(pred):
    """"(C) 0.2", "\\text{(C) } x", "\\mathrm{(C) \\ }0.2", "\\textbf{(C)}\\ 2" -> ("C", "0.2"); None without a label."""
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox)\s*\{([^{}]*)\}", r"\1", pred)
    m = LABEL.match(s)
    return (m.group(1), m.group(2).strip()) if m else None


def clock_seconds(s):
    """"8:34 p.m." / "20:34" / "8\\!:\\!34 \\text{ p.m.}" / "4:51:06 p.m." -> seconds after midnight; None if not a time or ambiguous."""
    s = strip_text(s).replace("\\!", "").replace("\\,", "").replace("\\ ", " ").replace("{:}", ":")
    m = re.fullmatch(r"\s*(\d{1,2})\s*:\s*(\d{2})(?:\s*:\s*(\d{2}))?\s*(?:\\text\{)?\s*([ap])?\.?\s*(?:m\.?)?\s*\}?\s*", s, re.I)
    if not m:
        return None
    h, mi, se, ap = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0), (m.group(4) or "").lower()
    if mi > 59 or se > 59 or h > 23:
        return None
    if ap:
        if not 1 <= h <= 12:
            return None
        h = h % 12 + (12 if ap == "p" else 0)
    elif 1 <= h <= 12:
        return None
    return (h * 60 + mi) * 60 + se


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
    m = re.fullmatch(r"[a-z](\\leq|\\geq|\\le|\\ge|<|>)(-?[\w\\{}.]+)", s)
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


def _same_text(a, b):
    n = lambda s: re.sub(r"[^a-z0-9]", "", strip_text(s).lower())
    return n(a) == n(b) or n(a) in n(b) or n(b) in n(a)


def lenient_equal(gold, pred_box, prompt):
    """Return (correct, fix). The live verify.py is tried first; fixes only run when it says no.
    A Box with .others (several different boxes in the answer channel) is correct only if every box is."""
    for b in getattr(pred_box, "others", ()):
        if not _lenient_equal1(gold, b, prompt)[0]:
            return False, None
    return _lenient_equal1(gold, str(pred_box), prompt)


def _lenient_equal1(gold, pred_box, prompt):
    if mv_equal(gold, pred_box):
        return True, None
    pred = inner(pred_box)
    if mv_equal(strip_text(gold), OPEN + strip_text(pred) + "}"):
        return True, "E1"
    gs, ps = letter_set(gold), letter_set(pred)
    if gs and ps and gs == ps:
        return True, "E2"
    opts = mc_options(prompt)
    lab = option_label(pred)
    if lab and lab[1]:  # "(X) value": the value must be the gold, and option X must be the gold when the options parse
        x, val = lab
        if mv_equal(gold, OPEN + strip_text(val) + "}") and (x not in opts or mv_equal(gold, OPEN + strip_text(opts[x]) + "}")):
            return True, "E3"
    if opts:
        ls = letter_set(pred)
        if ls and len(ls) == 1:
            p = opts.get(next(iter(ls)))
            if p and mv_equal(gold, OPEN + strip_text(p) + "}"):
                return True, "E3"
    gl = letter_set(gold)
    if lab and gl and len(gl) == 1 and lab[0] in gl:  # gold "(D)", reply "(D) a non-horizontal line"
        if not opts or lab[0] not in opts or not lab[1] or _same_text(lab[1], opts[lab[0]]):
            return True, "E8"
    tg, tp = clock_seconds(gold), clock_seconds(pred)
    if tg is not None and tg == tp:
        return True, "E9"
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
