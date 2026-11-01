# Verifier follow-up: v1 false positives, v2 fix, and miner labels (2026-11-01)

The gate probe has run our grader fix since 2026-10-30 (`VERIFIER_FIX = 'allan-lenient-v1-786cdb9'`, `_ALLAN_SRC` in `probe_worker.py`, `miner/validator/probe/allan_src.py`). Thanks for shipping it. `VERIFIER.md` said each rule needed unit tests with deliberately wrong answers before deployment. Those tests are now written. They found **5 ways v1 accepts a wrong answer** and **7 correct answers it still misses**. `code/lenient_grader_v2.py` fixes all 12 and is a drop-in replacement for `allan_src.py`.

## 1. What the tests found in v1 (`code/test_lenient_grader.py`, 50 cases)

**Wrong answers v1 scores as correct:**

| Case | Reply (gold) | Why v1 accepts it |
|---|---|---|
| Hedge | `$\boxed{3}$ or maybe $\boxed{5}$` (5) | P2 grades only the last box |
| Boxes disagree, no `<think>` | `Try \boxed{3}? No. Final: \boxed{5}` (5) | Same |
| Tail retracts | `$\boxed{5}$. Wait, that is wrong; the answer is 7.` (5) | P2 ignores what follows the box |
| Tail gives another answer | `$\boxed{5}$, but actually it is 7.` (5) | Same |
| Wrong option label | `\text{(B)}\ 119` when 119 is option C (119) | E3 drops the label and checks only the value |

A model can learn the hedge. The gate scores the candidate with this grader, so a head that boxes two answers gets credit whenever the last one is right.

**Correct answers v1 still scores 0:**

| Case | Reply (gold) | Rule in v2 |
|---|---|---|
| Option label in `\mathrm` | `\mathrm{(C) \ }0.2` (`.2`) | E3 now reads `\text`, `\textbf`, `\mathrm` and `\mbox` labels |
| Option label, no parsed options | `\text{(C) } \frac{1}{7}\triangle ABC` | Same |
| `\leq` / `\geq` | `x \leq 3` (`(-\infty,3]`) | E5: the alternation matched `\le` first and left `q3`. Longest operator first now |
| Gold is a letter, reply is letter + option text | `\text{(D) a non-horizontal line}` (`\text{(D)}`, task 3940) | E8 (new) |
| Same, with a formula | `\text{(C) } p + s = 0` (`\text{(C)}`, task 4573) | E8 |
| 24-hour clock | `20:34` (`8\!:\!34 \text{ p.m.}`, task 5442) | E9 (new) |
| 24-hour clock with seconds | `16:51:06` (`4:51:06 p.m.`, task 4713) | E9 |

## 2. What changes in v2

Same entry points and signatures: `lenient_project(text) -> (box, violation, fix)` and `lenient_equal(gold, box, prompt) -> (ok, fix)`. You can exec it exactly as `_ALLAN_SRC` is now (we tested it that way).

- **P2, several boxes:** if the answer channel holds more than one *different* box, the reply is correct only if *every* box is correct. A hedge fails. `\boxed{3\sec\frac12\theta}, which is choice (D). \boxed{D}` still passes, because both boxes are correct. The other boxes travel on the returned box (`Box(str)` with `.others`), so the probe code does not change.
- **P2, text after the box:** the reply is not graded if that text takes the answer back: `wait`, `actually`, `however`, `but`, `wrong`, `incorrect`, `mistake`, `instead`, `correction`, `recheck`, or another `\boxed`. A period or a unit phrase is still fine (`there are $\boxed{6}$ entries that are multiples of $67$.`).
- **E3:** for a reply of the form "(X) value", the value has to equal the gold, **and** option X has to be the gold whenever the options parse. Options written one per line as `(A) text` now parse too.
- **E5:** operator order fixed.
- **E8 (new):** the gold is a single option letter, and the reply is that letter plus text. If the options parse, the text has to match option X.
- **E9 (new):** clock times. Both sides have to be unambiguous: a.m./p.m. given, or the hour is 0 or ≥ 13. So `8:34` for `8:34 p.m.` is not accepted by this rule.

## 3. Regression on real replies (`code/regress_v2.py`, `results/verifier/regression_v2.json`)

We ran 6,558 finished replies through the probe logic (native grade first, then the lenient layer only if the native grade is not positive), once with v1 and once with v2. They are the 4,073 finished replies in the K=1 documents e196–e260, plus 2,485 replies from the 11 public free-file GREEN files e354–e413. The second set leaves out rollouts the miner labelled −1 (unfinished), which never reach the lenient layer.

| Change v1 → v2 | Replies | What they are |
|---|---|---|
| unchanged | 6,537 | |
| wrong → correct | 19 | E9 task 5442 ×5, E8 tasks 3940 ×8 / 4573 / 3830, E3 tasks 3094 ×2 / 3140 ×2. We checked each by hand, and all are correct. |
| correct → unresolved | 2 | Task 2345: an intermediate `\boxed{18}` in the answer channel before `\boxed{73}`. Task 1548: a degenerate reply with a malformed second box `\boxed{[-3,-frac{4}{3})}`. Both break "exactly one box" with a box that is not the answer. |

## 4. A false positive the lenient layer cannot fix: `verify.py` and clock times (fix: `code/clock_precheck.py`)

math-verify 0.9.0 parses `8:34` as a ratio and drops `\text{ p.m.}`. So the native grade accepts `8:34` and `8:34 a.m.` for the gold `8\!:\!34 \text{ p.m.}`. The lenient layer runs only when the native grade is *not* positive, so it never sees these. Only two pool tasks have clock-time golds, 4713 and 5442.

This happens in real data:
- **e413 (GREEN):** the group for task 5442 has `\boxed{8:34 \text{ AM}}` labelled **+1** and four `\boxed{20:34}` labelled **0**. 1,234 minutes after midnight is 20:34, so that group trains the head toward the wrong answer.
- **e215 (K=1):** two `8:34 AM` replies were graded positive, and `20:34` was graded negative.

**Fix, run in front of `native()`:** `clock_verdict(gold, box, prompt)` in `code/clock_precheck.py` (only uses `re`).
- It returns `None` when the gold is not a clock time, so every other task is untouched.
- Otherwise it compares times. A reply time is unambiguous if it has a.m./p.m. or its hour is 0 or ≥ 13. A bare 1–12 o'clock time counts only when the prompt asks for no am/pm. Task 4713 says *"Enter the time in the format "HH:MM:SS", without including "am" or "pm""*, even though its gold is written `4:51:06 p.m.`. In that case the reply is compared on a 12-hour clock.
- The docstring shows where it plugs into `grade()`: on the projected box before `native()`, and on the lenient box before `lenient_equal`.

**Tests:** `code/test_clock_precheck.py` has 27 unit cases, and `--tasks/--docs/--ff` replays every real reply to the two tasks. On the 17 real finished replies, exactly three grades change: `8:34 AM` goes correct → wrong in e413 ×1 and e215 ×2. Two behaviour changes are intended:
- On 5442, a bare `8:34` or `08:34` (no a.m./p.m.) becomes wrong.
- On 4713, `4:51:06` without am/pm stays correct, as its prompt asks.

## 5. Miner labels disagree with the gate

Six GREEN free files (e381, e383, e393, e398, e401, e413) use reward +1 = correct, 0 = finished but wrong, −1 = unfinished. Their 0 / +1 labels match the **native** grader exactly, so these miners label with `verify.py` and not with `miner/kit/grade.py`. In total, **156 rollouts labelled 0 are correct under the gate's grader (v1)**:

| Rollouts | Gate grader says correct because of |
|---|---|
| 99 | P2: protocol slips (83 text after the box, 13 a missing or extra `</think>`, 3 two boxes) |
| 17 | P1: the projection bug (`$$...$$` followed by a bare box) |
| 40 | E1 / E3 / E5 / E6 / E7: equivalent forms (15 tasks) |

Every one of them sits in a group with a +1 rollout, so the trainer pushes it down. Their advantages add up to −59, about 7% of all negative advantage in those six files. The gate then scores the same replies as correct. Under the batch-file rules the reward is the miner's, so the protocol cannot fix this. Two things would help:

- Say in `llms.txt` / `MINER_QUICKSTART.md` that `miner/kit/grade.py` is the gate's grader and the one to use for labels.
- Optionally, have `ff_format` / `status.json` report (without re-grading anything) how many rollouts with reward ≤ 0 the gate's grader scores as correct.

## Reproduce

```sh
cd code
/tmp/mv090/bin/python -I test_lenient_grader.py            # 50 cases, v1 vs v2; exits 1 if v2 gets any wrong
/tmp/mv090/bin/python -I regress_v2.py --tasks original-math7496.tasks.json \
    --docs /data/k1/docs --ff /data/k1/turns/{354,356,358,360,368,381,383,393,398,401,413}/batch.json \
    --out ../results/verifier/regression_v2.json
```

`lenient_grader_v2.py` sha256: see the commit. Same pinned dependencies as `VERIFIER.md`.
