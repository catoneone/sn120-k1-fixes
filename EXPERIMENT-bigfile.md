# Experiment: can one large file make a step the gate can see?

**Question.** A file is capped at 1,000 rollouts and trained as one step. On the current head that step is smaller than the gate's noise:
- The last 11 kings (e834 to e1049) score 0.876 ± 0.006 on AIME 22–26. That spread is what resampling alone gives, so the model has not moved.
- On the last four heads, 82 judged turns have mean z −0.11, and 9.8% pass. A file with no effect passes 16% of the time.

The proposal to test: **allow one large file (here about 8,000 rollouts) and train it as several small steps, then gate once.** One miner, one candidate and one payee, as today.

This experiment answers three things on the current head:
1. Does 8,000 rollouts of the head's own samples give a step the gate can see (z ≥ 2)?
2. Does the gain come from **more data** (a better direction) or from **several steps** (the gradient is recomputed along the way)? If it is the data, raising the file cap is enough. If it is the steps, the trainer has to loop.
3. How much do independent 1,000-rollout files agree on the direction? That tells us how large a file has to be before its direction is mostly signal.

It replaces the earlier one-step-vs-chain request (`EXPERIMENT-ff-steps.md`), which never ran. It is simpler: the live step already uses the fp32 master, support renorm and the gradient direction, so there are no recipe arms.

Nothing here goes on chain. No file is submitted, no candidate is promoted and nobody is paid.

## Arms

All arms start from the same head H and its fp32 master. The data is 496 tasks × 16 rollouts sampled from H, split into 8 files of 62 tasks (992 rollouts each). Each file is a valid file under today's caps.

| Arm | Data | How it is built | What it tells us |
|---|---|---|---|
| `S1` | file 1 (992 rollouts) | one step at the default lr 5e-7 | today's rule |
| `B1` | all 8 files (7,936 rollouts) | master(H) + (1/√8) × Σ (master(S_k) − master(H)), where S_k is file k trained as one default step from H. No extra training | large file, merged into one step |
| `C8` | all 8 files | chain of 8 steps: step k trains file k at lr 5e-7/√8 = 1.7678e-7, starting from the candidate of step k−1 | large file, several small steps |

**Why 1/√8.** The step has a fixed length per tensor, whatever the gradient is. Eight default-length steps in a row would be eight ungated turns, and live turns of this kind are slightly harmful on average. With 1/√8 per step, the chain travels about one default step in total when the eight directions are nearly independent.

**Why `B1` needs no training.** Below the default lr every tensor's step shrinks by exactly lr / 5e-7 (llms.txt §4). So (1/√8) × (master(S_k) − master(H)) is file k's step at lr 1.7678e-7 from H. `B1` adds the eight of them, all computed at H. `C8` uses the same eight files and the same step size, but computes each gradient where the previous step ended. So:
- `B1` vs `S1` is the value of 8× the data;
- `C8` vs `B1` is the value of recomputing the gradient along the way.

## Steps

### 0. Fix the head and the gate sample
- H = the current head: its bf16 checkpoint and its published fp32 master.
- G = the standing gate sample of H's current round, with the head probe that already exists for it (3 × 4 rollouts per task).

If H is promoted away during the run, keep going with the same H and G.

### 1. Pick 496 training tasks
- From gate-pool-v2: tasks where the head's pass rate is between 0.1 and 0.9, with at least 8 head rollouts behind that estimate (your prior, or recent head probes).
- **Leave out every task in G.** The gate in step 6 then has no trained task in it, and it runs unchanged.
- Draw 496 of the remaining tasks uniformly at random with a recorded seed. If fewer than 496 qualify, widen the range to 0.05–0.95.

### 2. Sample 16 rollouts per task from H
- The trainer's rendering of the batch prompt, as `kit/generate.py` does it.
- Gate sampling settings: temperature 1.0, top_p 0.95, top_k 20, thinking on. `max_tokens` 32,768, the training cap.
- Grade every rollout with the gate grader.

About 7,936 rollouts and 130M output tokens.

### 3. Build the files
- Shuffle the 496 tasks with a recorded seed and cut them into 8 lists of 62. File k holds all 16 rollouts of its 62 tasks.
- Reward: 1.0 if the gate grader says correct, 0.0 otherwise (wrong, capped or no final answer). No length term.
- **Keep every rollout.** Do not filter on `reencode_ok`: on head eec0079e that filter dropped 79% of the correct long replies. If a reply re-tokenizes to more than 32,768 tokens, trim it to the cap.
- Write each file twice, with the same rollouts and rewards:
  - `b1.json` … `b8.json`: `"train": {"adv_center": "mean"}`, default lr.
  - `s1.json` … `s8.json`: `"train": {"adv_center": "mean", "lr": 1.7678e-7}`.
- Run `kit/check_batch.py` on all 16.

### 4. Eight single steps from H
For k = 1 … 8, train `b{k}.json` as one step from H and its master. Use the live trainer host, or the kit:

```bash
python3 kit/train_step.py --head H --master H-fp32-master --batch b$k.json --hotkey $HK --out S$k --activation-checkpointing --report train_S$k.json
```

Keep each candidate's fp32 master. `S1` is the first of them.

### 5. Direction agreement and `B1`
```bash
python3 code/merge_delta.py cos   --head-master H-fp32-master --cands S1 S2 S3 S4 S5 S6 S7 S8 --out delta_cos.json
python3 code/merge_delta.py merge --head-master H-fp32-master --cands S1 S2 S3 S4 S5 S6 S7 S8 --scale 0.35355339 --head H --out B1
```
- `cos` writes the 8 × 8 cosine matrix of the updates, overall and per block of layers.
- `merge` writes the `B1` checkpoint (bf16 = round-to-nearest-even of the new master) and its fp32 master.

Both need about 10 × the largest tensor of host RAM, and `merge` also holds the new master (36 GB) while writing it.

### 6. The chain `C8`
C_0 = H. For k = 1 … 8, train `s{k}.json` as one step from C_{k−1} and the master that step k−1 wrote. This is the same hand-over the live chain does after a GREEN.

If you use `kit/train_step.py`: it refuses a master whose `FP32-MASTER.json` names another head, so write that tag for each intermediate candidate.

Then check the chain against the single steps:
```bash
python3 code/merge_delta.py chain --head-master H-fp32-master --singles S1 S2 S3 S4 S5 S6 S7 S8 --chain C1 C2 C3 C4 C5 C6 C7 C8 --scale 0.35355339 --out chain_check.json
```
**Built-in check:** step 1 of the chain starts from H, so it must equal (1/√8) × file 1's default step. `chain_check.json` should show cos ≈ 1 and rel_diff ≈ 0 for step 1. If it does not, a lower lr does not scale the step exactly, `B1` is not what the table says, and I need to know before the gate runs.

### 7. Gate
Score `S1`, `B1` and `C8` on G against H's existing probe, with the live gate unchanged: 8 rollouts per task, 65,536 output tokens, the same grader and the same estimate. These are shadow runs, not turns.

### 8. Optional
Run bench-v2 AIME 22–26 on whichever of `B1` and `C8` has the higher gate z.

## Decision rules (written before any results)

An arm **works** if its gate z is at least 2, the cap-hit rule passes, and on the tasks the head solves 12/12 it loses less than 1pp.

| Result | Reading | What we would propose |
|---|---|---|
| `B1` or `C8` works | A file of about 8,000 rollouts makes a step the gate can see | Raise the file cap. Add multi-step training only if `C8` beats `B1` by z ≥ 2. Together with: leave the file's own tasks out of the gate, regrade a sample of the file's rewards, and raise the bar to z > 2 |
| Both are positive but below z 2 | Not enough data at this size | Use `delta_cos.json` to estimate the file size needed, then decide on a larger run |
| Both have z < 1 | On this head, GRPO on the head's own samples does not give a visible step even at 8,000 rollouts | Drop the proposal. The limit is new information (harder tasks, a stronger teacher), not step size |
| `B1` works, `C8` is clearly worse | The data goes stale along the chain, or noise adds up | Large file as one merged step only |
| `C8` works, `B1` does not | Recomputing the gradient is what matters | Multi-step training, with the number of steps capped by the off-support fraction |

**Predictions:**
- `S1`: z between −1.5 and +0.5.
- `B1`: z between +1 and +3.
- `C8`: within 1.5 z of `B1`.
- Mean pairwise cosine of the eight updates: 0.01–0.05. This one is a guess.

## Cost
- **Sampling:** about 130M output tokens once.
- **Training:** 16 one-file steps (8 single, 8 chained), each the size of a normal turn. The 8 single steps can run in parallel; the chain is sequential.
- **Gate:** 3 candidate probes on G. The head probe is reused.
- **`merge_delta.py`:** CPU only, a few minutes per command on local disk.

## What to send back
- The seeds, the 496 task ids and the 8 task lists.
- The 16 files, or their sha256 if you prefer to keep them.
- For each of the 16 training steps: the trainer report (pre-clip gradient norm, update norm, `support_renorm` outside fraction).
- `delta_cos.json`, `B1/MERGE.json` and `chain_check.json`.
- For `S1`, `B1` and `C8`: the gate result (Δ, SE, z, cap-hit, per env) and the per-task grades of head and candidate, in the format of a GREEN reveal. I will compute the arm-vs-arm differences and the split by head pass rate from those.
- The AIME row, if step 8 ran.

## What this does not test
- **Files ten times larger again.** That can only be extrapolated from the cosines.
- **Overfitting the public pool and forged rewards.** More steps amplify both. They are rule questions; this run avoids them by keeping the training tasks out of G and grading its own rollouts.
- **All-positive or all-negative files** (`adv_center: none`). Those change sharpness and length, which would confound the comparison.

## Tested so far
`code/test_merge_delta.py` runs `merge_delta.py` on CPU with small synthetic masters (sharded head master, single-file candidate masters, a bf16 head with one non-trainable tensor):
- `cos` against cosines computed directly;
- `merge` against the expected master, its bf16 rounding and the untouched non-trainable tensor;
- `chain` on a chain built to equal the scaled single steps (rel_diff 1e-6), and on a perturbed chain (step 1 matches, later steps do not).

Nothing here has run on a GPU or on real masters.
