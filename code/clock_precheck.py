"""Clock-time check to run IN FRONT OF the native grade (verify.py / native()).

math-verify 0.9.0 reads `8:34` as a ratio and drops `\\text{ p.m.}`, so the native grade accepts `8:34 a.m.` for the gold
`8\\!:\\!34 \\text{ p.m.}`. The lenient layer only runs after a non-positive native grade, so it cannot catch this.
In the 7,496-task pool only tasks 4713 and 5442 have clock-time golds.

  clock_verdict(gold, box, prompt) -> None   the gold is not a clock time: grade as before, nothing changes
                                    -> True   the box is the same time
                                    -> False  anything else (another time, an ambiguous time, not a time)

A time in the reply is unambiguous if it has a.m./p.m. or its hour is 0 or >= 13. A bare 1-12 o'clock time is accepted only
when the prompt asks for the time without am/pm (task 4713: "Enter the time in the format "HH:MM:SS", without including
"am" or "pm"."), and then compared on a 12-hour clock.

Use (miner/kit/grade.py grade(); probe_worker.py has the same shape):

    box = project(out)
    cv = clock_verdict(gold, box, prompt) if box is not None else None
    if cv is not None:
        return ('positive' if cv else 'negative'), 'clock'
    ... native(), then the lenient layer; there, after gbox = lenient_project(text):
    cv = clock_verdict(gold, gbox, prompt)
    if cv is not None: ok2, efix = cv, 'clock'   # instead of lenient_equal

Self-contained (only `re`); `/tmp/mv090/bin/python -I test_clock_precheck.py` runs the tests.
"""
import re

OPEN = "\\boxed{"
TIME = re.compile(r"\s*(\d{1,2})\s*:\s*(\d{2})(?:\s*:\s*(\d{2}))?\s*(?:([ap])\.?\s*m\.?)?\s*", re.I)
NO_AMPM = re.compile(r"without\s+(?:including\s+)?[\"'`]?a\.?m\.?[\"'`]?\s+or\s+[\"'`]?p\.?m\.?", re.I)


def _clean(s):
    s = s.strip()
    if s.startswith(OPEN) and s.endswith("}"):
        s = s[len(OPEN):-1]
    for _ in range(3):
        s = re.sub(r"\\(?:text|textbf|mathrm|mbox)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("{:}", ":").replace("\\!", "").replace("\\,", " ").replace("\\ ", " ").replace("~", " ").replace("$", "")
    return s.strip()


def parse_clock(s):
    """-> (hour, minute, second, 'a' | 'p' | '') or None."""
    m = TIME.fullmatch(_clean(s))
    if not m:
        return None
    h, mi, se, ap = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0), (m.group(4) or "").lower()
    if mi > 59 or se > 59 or h > 23 or (ap and not 1 <= h <= 12):
        return None
    return h, mi, se, ap


def _seconds(h, mi, se, ap):
    if ap:
        h = h % 12 + (12 if ap == "p" else 0)
    return (h * 60 + mi) * 60 + se


def clock_verdict(gold, box, prompt=""):
    g = parse_clock(gold)
    if g is None:
        return None
    p = parse_clock(box) if box else None
    if p is None:
        return False
    if g[3] == "" and 1 <= g[0] <= 12:  # an ambiguous gold: compare on a 12-hour clock
        return (g[0] % 12, g[1], g[2]) == (p[0] % 12, p[1], p[2])
    gs = _seconds(*g)
    if p[3] or p[0] == 0 or p[0] >= 13:
        return _seconds(*p) == gs
    if NO_AMPM.search(prompt or ""):  # the prompt asks for no am/pm: a bare 1-12 o'clock time is read on a 12-hour clock
        return (p[0] % 12, p[1], p[2]) == ((gs // 3600) % 12, (gs // 60) % 60, gs % 60)
    return False
