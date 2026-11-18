# Pool audit, part 2: the DAPO tasks added in v2 (2026-11-18)

The v2 pool (`math_pool_v2.jsonl`, sha256 `8d34d276…`) has 2,269 DAPO tasks that v1 did not have: ids `dapo:11813` to
`dapo:14081`. v1 only took the 11,813 DAPO tasks with integer answers; these are the rest. Our 11-08 audit
(`POOL-AUDIT.md`) mostly used v1 gate data, so it reached few of them.

We reviewed all 2,269. **1,034 of them (46%) are bad: wrong key, ambiguous, or broken.** Another 135 have a correct key in a
format a normal answer may fail. The lists are in `data/pool_audit_v2new_lists.json`: **783 to drop and 387 to re-key.**

These tasks matter more than their share of the pool suggests:
- **Gate.** Since r020, **42% of the head's failures in the gate land on bad tasks.** Without them the head's weighted pass
  rate would be 0.921 instead of 0.875. So about 4.6pp of the "remaining gap" is not ability.
- **Training.** Miners pick tasks the head got wrong in earlier rounds, so they pick bad tasks more and more often. **In the
  last two winning files, half of the rows that train are on bad tasks** (r042 47%, r046 50%; it was 2–28% before r018).
  On a bad task, a reply that happens to match a wrong key is trained up, and a correct reply is trained down.

We think this is one reason the gains got smaller after r020: as the head improves, a larger share of what is left to fix
is label noise.

## How we reviewed them
- Reviewer: an OpenAI model through Codex (`gpt-6.1-sol`, high reasoning effort) in a read-only sandbox, so it could run Python to
  brute-force or check numerically. 20 tasks per call.
- Evidence per task: the problem, the key, the head's gate results in every round since r001, the final answers in public
  winning files with their grades, answers from two stronger open models where we had them (Qwen3.6-27B, Qwen3.5-27B), and
  the 11-08 verdict if any. It was told to solve each task itself and not to trust the evidence.
- Same five verdicts as on 11-08: keep, bad key (with the correct answer), key format (with a clean key), ambiguous, broken.

| verdict | tasks | head's mean gate pass |
| --- | --- | --- |
| keep | 1,099 (48%) | 0.55 |
| bad key | 252 (11%) | 0.21 |
| ambiguous | 519 (23%) | 0.35 |
| broken | 264 (12%) | 0.34 |
| key format | 135 (6%) | 0.46 |

- 2,240 of the 2,269 verdicts are high confidence. 29 are medium or low.
- The harder the task is for the head, the more likely it is bad. Of the 520 tasks the head almost never solves, about 72%
  are bad.

Typical problems:
- **Rounded or cut keys.** dapo:12533 asks for an average difference: the exact answer is −25/7, the key is −3.57, so
  −25/7 is graded wrong. dapo:12746: the answer is 63 rubles 66 kopecks, the key is 63.
- **One key for several questions.** dapo:13499 asks to split 46 rubles into 8 parts; the key is the first part, 4.
- **A number as the key of a yes/no or construction task.** dapo:12830 (does such a number exist? a) no, b) yes) has key
  111111111; dapo:12235 ("cut a 3×9 rectangle into 8 squares") has key 8.
- **Missing figure or table, or a fragment of a solution as the problem** (dapo:12216, dapo:12226).
- **Wrong keys** that the reviewer proved by brute force or by a full argument, e.g. dapo:11859 (the key is 3; the maximum is
  1335, by a bound from the exterior angles and a construction that attains it).

## Checks
- **Blind second review.** We drew 60 tasks at random (40 flagged, 20 kept), removed the 11-08 verdicts, and ran the review
  again. Same verdict for 57 of 60; same keep-vs-act for 59 of 60. Both runs called 8 tasks "bad key" and gave the same
  corrected answer for all 8. Both runs used the same model, so this shows the verdicts are stable, not that a second model
  agrees.
- **Against the 11-08 audit** (134 tasks in common). Of its 43 "ambiguous", 38 are ambiguous here too. Of its 36 "broken", 35
  are flagged here too (broken, ambiguous, or bad key). The main difference is 21 keys of the form `k = 1012`: on 11-08 we
  called them key format under the v1 grader; the v2 grader accepts `1012` for them, so they are kept here. The value is the
  same in both.
- **Against tasks we checked by hand** in the r046 winning file: the reviewer flags 13 of the 14 that are in this range. The
  14th (dapo:12136, `3:30 \text{ p.m.}` graded wrong) has a correct key; that is a grader issue, and verifier v3 handles it.

## Impact, in detail
"Bad" below means the 1,034 tasks above plus the 549 tasks of the 11-08 lists that are in other parts of the pool: 1,450
tasks, 6.7% of the pool.

Gate, r020 to r046 (26 rounds, 51,997 drawn tasks):
- The head failed 10,192 times; 4,259 of those (41.8%) were on bad tasks, 40.9% by IRT weight.
- The head's weighted pass rate is 0.875; without the bad tasks it is 0.921.
- Bad tasks are almost always 0 for the head, so π is small and the weight 1/π is large when one is drawn. They add
  variance to every delta.

Training, all public winning files, counting only groups of pool tasks with mixed rewards (the rows that train):

| winning file | groups that train | on bad tasks | share of training rows on bad tasks |
| --- | --- | --- | --- |
| r005 | 250 | 17 | 7% |
| r010 | 250 | 4 | 2% |
| r011 (same file as r008, r014) | 245 | 68 | 28% |
| r018 | 115 | 55 | 48% |
| r027 | 120 | 34 | 28% |
| r035 | 250 | 103 | 41% |
| r042 | 405 | 138 | 47% |
| r046 | 287 | 143 | 50% |
| all files | | | 29% (2,633 of 8,936 rows) |

## What we suggest
1. **Drop the 783 ambiguous and broken tasks**, together with the 11-08 drop list. This only removes noise.
2. **Re-key the 387 bad-key and key-format tasks**, or drop them for now. `new_key` in the list is the reviewer's answer; 13 of
   them are not high confidence and are better dropped. As on 11-08, a new key should first be shown to be reachable by
   the head.
3. **Ship verifier v3** (`VERIFIER-v3.md`) for the notation cases (times, lists of values).
4. Until the pool is cleaned, miners who pick "tasks the head got wrong" will keep picking these. Publishing the lists lets
   honest miners filter them out.

## Limits
- We only reviewed this range. The other 11,813 DAPO tasks and the 7,492 MATH tasks were covered only by the 11-08 audit,
  which reviewed 1,020 tasks chosen from gate data. The head passes old DAPO at 0.82, so bad tasks should be much rarer
  there, but not absent.
- "Ambiguous" is applied strictly. Some of these tasks have a reading most people would take. If you want to keep more
  tasks, drop only the high-confidence ones, or have a second model look at the ambiguous group.

## Files
- `data/pool_audit_v2new_lists.json`: `remove` (783: id, verdict, confidence, key, reason) and `fix_key` (387: id, verdict,
  confidence, key, new_key, reason).
- The full per-task verdicts with the reviewer's reasons, and the scripts, are available on request.
