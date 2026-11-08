# Gate pool audit: 549 tasks to drop or re-key (gate-pool-v2, 2026-11-08)

`miner/data/gate-pool-v2.jsonl` (sha256 `8d34d276…`, 21,574 tasks: 7,492 affine_math and 14,082 dapo) serves two
purposes. It is the gate's evaluation pool, and it is where miners get their training tasks. We found **549 tasks (2.5% of
the pool) that should be dropped or re-keyed**. Almost all of them are DAPO. The lists are in
`data/pool_audit_lists.json`.

This is a pool decision, separate from verifier v3. We suggest you drop tasks first and re-key them later. Dropping a task
only removes noise. A new key should first be shown to be reachable by the head.

## How we found them
1. **Gate results.** 42 closed rounds publish per-task head and candidate results. Together they cover 11,487 tasks and
   416K samples. 631 tasks were never solved in 20 or more attempts.
2. **Miner replies.** The 64 public GREEN files from e354 to e903 hold 30,470 replies. We took the final answers per task and
   graded them with the live grader and with v2.
3. **Rules.** Tier A (1,020 tasks) holds the strong signals:
   - never solved, or solved less than 5% of the time
   - the miners' consensus answer differs from the key
   - a multi-part question keyed to one part
   - a key containing an assignment, "or", words, or a trailing period
   - DAPO rewrite leftovers
   - normalisation-dependent asks such as "sum of the coefficients of the ellipse equation"
   - duplicate problems with conflicting keys
4. **Per-task review.** Claude Opus checked every Tier A task. It solved or brute-forced each one where feasible and gave
   one of five verdicts: keep, bad key (with the correct answer), ambiguous, broken, or key format (with a clean key).
5. **Checks.**
   - **Blind second review of 60 tasks.** The second reviewer did not see the first verdict. The two reviews agree on
     keep vs. act for 55 of 60 tasks, and on the exact verdict for 52.
     - Of the 40 tasks the first review flagged, the blind review confirmed 39.
     - Both reviews called 12 tasks "bad key", and they gave the same corrected answer for every one.
     - The 5 disagreements are borderline: key notation that the live grader already accepts, and two tasks that depend on
       reading the problem literally or as intended.
   - **Agreement with the gate data.**
     - Of the 213 bad-key verdicts, 190 are on tasks that were never solved and 21 on tasks solved less than 5% of the
       time.
     - Of the 81 reviewed tasks the head solves at least 50% of the time, none were called bad key or broken.
   - **Brute force by hand.** We recomputed two tasks by brute force, and both match the review:
     - dapo:11573 has 12 integer pairs, not 0.
     - dapo:319 has 970, not 1010.

## Result (Tier A, 1,020 tasks)
| Verdict | Tasks | Action |
|---|---|---|
| keep | 471 | — |
| bad key | 213 | re-key the 134 rated high confidence; drop the other 79 |
| broken | 159 | drop |
| ambiguous | 87 | drop |
| key format | 90 | re-key with the clean form |

In total that is **325 tasks to drop** (317 dapo) and **224 to re-key** (179 dapo). Of the 631 never-solved tasks, 401 are
in this list. The other 230 are hard but valid, so "never solved" alone is not a reason to drop a task.

Typical cases:
- A DAPO rewrite template that doesn't fit the original problem. Examples: "the answer is m/n, give m+n" applied to an
  answer that isn't a fraction, or "now let n=…" against the problem's own conditions.
- An orphaned part (2), or a multi-part question with one key.
- "Find all polynomials" keyed to a single number.
- A squared modulus given as the key (dapo:10399: the key is 5, the answer is √5).
- Wrong keys that brute force refutes (dapo:11435: the key is 2016, the answer is 1512).

## Why it matters
- **Gate.** The head scores almost 0 on these tasks, so they lower the ceiling without measuring anything. Their P is near
  0, so their π is small too. When one is drawn, its 1/π weight adds variance and no signal. Our rough estimate, without
  the private π, is that the ceiling of the pool-weighted pass rate rises by 1–2pp.
- **Training.** Miners label their data with the gate grader. On a task with a wrong key, a correct answer becomes a
  negative example.

## Not covered
- Tier B (804 weaker-signal tasks) and the remaining ~19.7K tasks were not reviewed.
- Most corrected keys come from one reviewer. Only the 12 overlapping bad-key tasks have a second, independent source.
