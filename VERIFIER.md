# The affine_math verifier scores correct answers as 0: evidence from the K=1 batches and a tested fix (2026-10-31)

## Summary

- The 58 K=1 batches from e196–e260 contain 544 groups and 4,352 rollouts, of which 4,073 finished with EOS. **481 of the finished rollouts are correct answers that the live grader scores 0.**
- **161 of the 544 groups (30%) would be all-correct under a fixed grader.** Their only "failures" are grading artefacts, so all of their training signal comes from the grader. The same few tasks keep coming back in turns from different coldkeys. For example, 6 of the 9 groups in e259 are such tasks.
- The gate uses the same rules. In the gate reveals for e196–e207 (95,932 samples), 1,020 samples (1.06%) are finished answers that the projection marks unresolved, which is 7.2% of all gate failures. The reveals don't include the text, so we can't say how many of them are correct.
- We reproduce the live grading exactly: 0 mismatches on the 4,073 finished rollouts. On top of it we add nine deterministic fixes. Together they recover all 481. We checked every one of the 290 wrong→correct changes by hand, and all are correct. The 489 genuinely wrong answers stay wrong, and no correct answer changes.

## How the live grader scores a reply

1. **Projection** (`subnet/mimo_answer_projection.py`). The reply must contain exactly one `<think>` and one `</think>`. The answer channel after `</think>` must contain exactly one `\boxed{}`. After the box, only the closing `$`, `$$`, `\]` or `\)` that matches an opener before the box may follow. If nothing follows the box, the text before it must not end in `$`, `\[` or `\(`. If any of these checks fails, the reply is `neutral` with reward 0, and `verify.py` never runs.
2. **`verify.py`** (math-verify 0.9.0). The gold answer is parsed as `\boxed{gold}` and the prediction is the last complete box. If they are equal the reward is 1, otherwise 0.

Every task's system prompt says: *"Solve the math problem. Reason step by step in plain text, then end your response with the final answer in `\boxed{}`. Emit exactly one `\boxed{}` block and nothing after it."*

## What goes wrong, and the fixes

| Problem | Example (task) | Fix | Recovered |
|---|---|---|---|
| **Projection bug** | A bare `\boxed{25200}` on its own line after `$$20 \times 30 \times 42 = 25,\!200$$` is rejected. The check sees the *closing* `$$` as an unclosed opener (1950, 1774, 5689, 5114). | **P1**: count unescaped `$` before the box. Reject only if the count is odd. | 34 |
| **Projection gives 0 for a protocol slip** | Trailing text such as `The answer is $\boxed{819}$.` or `the graph has $\boxed{3}$ distinct $x$-intercepts.`; more than one box; a reply that skips `<think>` (task 107: the head's first token is `3` with probability ≈ 0.41); two `</think>`. | **P2**: grade the last complete box as usual and record a format violation, to be charged as a small separate penalty. | 157 (+17 now graded wrong instead of unresolved) |
| **Wrappers, separators** | The 3439 prompt says *enter "B, E" without the quotation marks*. The model writes `A, D, F, G, H`, the gold is `\text{A, D, F, G, H}`, and the answer is scored wrong. The same happens on 6119, 4456, 6383, 5565 and 4876. 4135 is the other way round: the gold is `D, E` and the model writes `\text{D, E}`. On 1905 only the thousands separators differ (`29,\!322,\!216`). | **E1**: strip `\text{}`, `\textbf{}`, `\mathrm{}` and thousands separators. | 81 |
| **Multiple choice** | On 5443 the gold is `119` and the model writes `C` (option C is 119). On 3138 the model writes `\text{(A)}\ \frac{bx}{h}(h-x)`. Also 3129, 3134, 5339, 3137. | **E3**: map a letter to the option text parsed from the prompt; drop an "(A)" label in front of a value. | 68 |
| **Vector vs tuple** | 7303 and 7009 ask to *"Find the point where the lines intersect"*. The model writes `(6, -17)`; the gold is a `pmatrix`. | **E4**: treat a pmatrix column vector and a tuple as the same. | 48 |
| **Inequality vs interval** | 6755 `a \neq 3`, 3520 `k < -\frac{1}{5}`, 4429 `2 < x < 4`. | **E5**: convert a single-variable inequality or `\neq` to an interval. | 26 |
| **Scalar identity** | On 7409 the model writes `-5\mathbf{I}` and the gold is an explicit 2×2 matrix. The problem doesn't even fix the dimension. | **E6**: expand `kI` to the gold's size. | 34 |
| **Complex numbers** | On 7163 the model writes `-32 + 32\sqrt{3}i` and the gold is `-32 + 32i\sqrt{3}`. | **E7**: treat `i` as the imaginary unit and check that the difference simplifies to 0. | 33 |

E2 (letter lists compared as sets) never fires on this data, because E1 already covers the letter lists here. P2 plus E3 together recover one more rollout, which brings the total to 481.

## Regression results (`results/verifier/regression.json`)

| Live label → after fixes | Rollouts |
|---|---|
| correct → correct | 3,066 |
| wrong → wrong | 489 |
| wrong → correct | 290 |
| unresolved → correct | 191 |
| unresolved → wrong | 19 |
| unresolved → unresolved (no complete box anywhere) | 18 |

- **Reproduction:** projection plus `verify.py` reproduce the live label on all 4,073 finished rollouts.
- **False positives:** we checked all 290 wrong→correct changes by hand, and all are correct. On 3134 and 5339, for example, the letter the model chose is the correct option. The 489 genuinely wrong answers are still graded wrong after the fixes, and no correct answer becomes wrong.
- **Format violations kept for the penalty (P2):** 192 in total. 75 have text after the box, 23 have more than one box, 91 have a malformed thinking channel and 3 really do leave a `$` open.
- **Caveat:** this only shows the fixes are clean on the data we have. The rules can over-accept forms they have not seen. Before deploying, each fix needs unit tests with deliberately wrong variants: a wrong letter, a wrong tuple, a wrong interval, a wrong complex number. E3 also depends on parsing the options from the prompt.

## What we suggest

1. **Fix the projection bug (P1).** This one involves no policy choice.
2. **Charge protocol slips as a small separate penalty instead of 0 (P2).** For example, `r = 1[correct] − 0.1·1[violation]`. A correct derivation that ends with a period then loses 0.1. Today it gets reward 0, and the negative advantage pushes down every token of the derivation.
3. **Accept equivalent answers (E1–E7),** or correct the reference answers of the affected tasks.
4. **Find the remaining cases from data.** You have the text of every probe. Flag tasks where most wrong answers are the same string, then review them by hand. Some of these are real systematic errors and must stay wrong; for example, on 5029 the model writes `4` 31 times while the gold is `8`. Add the genuine alternatives to a versioned, signed list of accepted answers.
5. **Switch the miner, the validator and the gate to the new grader version at the same epoch boundary.** A miner's claimed outcome has to match the validator's grade, otherwise the group is rejected.

## Reproduce

```sh
uv venv -p 3.12 /tmp/mv090
VIRTUAL_ENV=/tmp/mv090 uv pip install math-verify==0.9.0 latex2sympy2-extended==1.11.0 \
    sympy==1.14.0 antlr4-python3-runtime==4.13.2 mpmath==1.3.0
cd code
python3 fetch_k1.py --out /data/k1 --epochs 196-260      # epochs without a turn are skipped
/tmp/mv090/bin/python -I lenient_grader.py \
    --tasks original-math7496.tasks.json --docs /data/k1/docs \
    --out ../results/verifier/regression.json
```

`original-math7496.tasks.json` is the pool from `assets/` (branch `validator/v829i-promotion-gate-20261026`). Every rule in `lenient_grader.py` sits in its own function, and a fix only runs when the live grade is not "correct". `regression.json` contains the scope, both label matrices, the count for each fix, and every changed (task, fix) pair with the answers.
