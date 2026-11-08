# Verifier v3: two more notation rules found by a pool audit (2026-11-08)

The gate probe runs `allan-lenient-v1-786cdb9`. `VERIFIER-v2.md` described v2 (hedges, retractions, option labels, clock
times). v3 is v2 plus two rules. Both were found by grading every public GREEN miner reply against the gate pool and listing
the replies that are right but still graded wrong. `code/lenient_grader_v3.py` is a drop-in replacement for `allan_src.py`,
with the same entry points.

## 1. What v3 adds

| Rule | Gold | Reply that v2 grades wrong | Found in |
|---|---|---|---|
| E10 matrix row spacing | `\begin{pmatrix} 8/3 \\ 4/3 \\ -4/3 \end{pmatrix}` | `\begin{pmatrix} \dfrac{8}{3} \\[4pt] \dfrac{4}{3} \\[4pt] -\dfrac{4}{3} \end{pmatrix}` | math:6800, 6881, 7110, 7149, 7362, 7454 |
| E10 unbraced `\frac` in the gold | `2007 + \frac\pi 2` | `2007+\frac{\pi}{2}` | math:6807; 74 pool golds use `\frac12`-style |
| E10 trailing punctuation in the gold | `30^\circ.` | `30^\circ` | math:2871; 10 pool golds end in `.` or `,` |
| E10 percent sign | `6.4` | `6.4\%` | math:2376; 38 pool golds carry `\%` |
| E10 `\phantom` | `\begin{pmatrix} 1 & -1 \\ 1 & \phantom -1 \end{pmatrix}` | `\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}` | math:6782 |
| E10 interval with `;` | `[-1;4]` | `[-1, 4]` | dapo:12899 |
| E11 roots as an unordered list | `x_{1}=1,x_{2}=-6` | `x=-6 \text{ or } x=1` | dapo:12701; 10 pool golds are root lists |

E10 rewrites only notation on both sides (row spacing, `\phantom`, unbraced `\frac`, `\%`, a `;` between interval ends,
trailing punctuation), then asks `verify.py` again. It never changes a value. E11 applies only when every assignment names
the same unknown (`x`, `x_1`, `x_2`). `A=5, B=10` names different unknowns, so it stays ordered.

Roots given as a set or with `\pm` need no new rule. The live `verify.py` already treats `\{-6, 1\}` and `-6, 1` as
equal to `x_{1}=1,x_{2}=-6`. It also treats `x=\pm 2`, `x=2 \text{ or } x=-2`, `\pm 2` and `2, -2` as all equal. And it
still rejects a set with a wrong root, a set missing a root, and `x=2` for `x=\pm 2`. Those cases are in the tests below, so
a later change cannot break them.

## 2. Tests (`code/test_lenient_grader_v3.py`)

- 16 right answers that must pass, and 17 wrong ones that must stay wrong. The wrong ones include a sign error inside a
  matrix, a missing root, a duplicated root, swapped values for different unknowns, a set with a wrong or missing root, one
  of two `\pm` roots, a percent that is off by 10×, `\frac{50\cdot2^{100}}{2^{100}-4}` for 50, and a digit swap. All 33
  pass.
- All 50 v2 cases give the same result under v3.

## 3. Regression on real replies (`code/regress_v3.py`)

There are 30,470 replies in the 64 public GREEN files from e354 to e903. We graded each one like the probe does: native
grade first, then the lenient layer.

```
/tmp/mv090/bin/python -I code/regress_v3.py --batches DIR --pool miner/data/gate-pool-v2.jsonl \
    --allan-src miner/validator/probe/allan_src.py --out results/verifier/regression_v3.json
```

`DIR` holds the files as `<epoch>.json`, copied from `/k1/turns/<epoch>/batch.json`. The inputs we used are listed in
`results/verifier/regression_v3_inputs.json`: the sha256 of each of the 64 files, gate-pool-v2 sha256 `8d34d276…`, and
`allan_src.py` at affine@7fea2e74. Every changed reply is in `results/verifier/regression_v3.json`.

| Change v2 → v3 | Replies |
|---|---|
| wrong → correct (E10, matrix row spacing) | 16 |
| wrong → correct (E10, `[-1;4]`) | 1 |
| wrong → correct (E11) | 1 |
| correct → wrong | 0 |

We checked every changed reply by hand, and all are correct. Example: `-\dfrac{10}{15}` for `-2/3`, and `\dfrac{4}{\sqrt2}`
for `2\sqrt2`.

## 4. A live false positive the lenient layer cannot fix

`verify.py` grades the gold `[-1;4]` as equal to `(-1, 4)`, so it does not tell open from closed intervals here. The lenient
layer runs only when the native grade is not positive, so it never sees this case. The fix belongs in the answer key: write
`[-1,4]`. The key-format list is in `data/pool-audit/key_notation.json`.
