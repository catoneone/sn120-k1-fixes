# Experiment: why 6b16e298 + e226–e235 dips and doesn't recover (head or data?)

**Goal.** With fp32 master + support renorm + gradient-direction step:
- the chains from 7fa0305a and from 1636928e dip and then recover;
- the chain from 6b16e298 over e226–e235 collapses to about 30% and stays there.

This experiment separates the two candidate causes, the start head and the batches. It also fills the gaps in the 6b16e298 runs:
- a 1× chain;
- our `stream` entry point vs. the `exp_k1.py` wrapper;
- a skip test;
- a gated run.

Everything runs with one command, `code/run_2x2.sh`. A whole run takes about 5–5.5 h on 8×H200, about 27 H200-hours in total.

## Background (from the public turn files)

**Every batch in the collapsing chain comes from one head.** All of e226–e236 were sampled from 6b16e298. So that chain is one head's data applied as 10 sub-steps, the same shape as the proposed per-miner large batch.

**Termination push.** For each batch this is:
- the sum of A_i/K_i over finished answers,
- divided by the sum of |A_i|/K_i over all answers.

Positive means the batch rewards finishing.

| Chain | Steps (termination push) | Result |
|---|---|---|
| 7fa0305a, 1× | e197 −0.03, e198 −0.13, e199 +0.30, e200 +0.10, e201 +0.50, e202 +0.29 | 68% at s3, then +2.4pp at s6 |
| 1636928e, 1× | e240 +0.48, e241 +0.33, e242 +0.45, e243 +0.46, e244 +0.57, e245 +0.05 | −15.4 at s1, then +4.1 at s3 |
| 6b16e298, 0.25× / 0.5× | e226 +0.34, e227 +0.13, e228 +0.20, **e229 −0.05, e230 −0.13, e231 −0.17**, e232 +0.28, e233 +0.17, e234 +0.38, e235 −0.03 | no recovery |

**Two readings fit the data so far:**
1. **e229–e231 deepen the dip.**
   - For: they are the only negative batches, and in the 0.25× chain the fall from 61.6% to 30.0% happens exactly over them.
   - Against: the 0.5× chain was already at 33% after e226–e228.
2. **Once the model is mostly capping, stale batches can't pull it back.** Those batches were sampled from a head that rarely caps, so they hold few capped answers to push down. The finish-favoring e232–e234 don't help either chain.

## Arms

All arms use fp32 master + support renorm + gradient-direction step (`--arm fp32_renorm --opt normgrad`). `--step-mult` scales each tensor's step, like the 0.5× and 0.25× arms of the `exp_k1.py` wrapper. Every arm is scored on the gate set against its own start head, paired per task.

| Arm | Head | Batches | Step | Scored at | Question |
|---|---|---|---|---|---|
| `A7fa_e197_1x` | 7fa0305a | e197–e206 | 1× | s1, 3, 6, 10 | Known case (dips, recovers); same setup as the other arms |
| `A7fa_e226_1x` | 7fa0305a | e226–e235 | 1× | s1, 3, 6, 10 | **Is it the data?** |
| `B6b_e197_1x` | 6b16e298 | e197–e206 | 1× | s1, 3, 6, 10 | **Is it the head?** |
| `B6b_e226_1x` | 6b16e298 | e226–e235 | 1× | s1, 3, 6, 10 | Known collapse, at 1× (not run yet) |
| `A7fa_e197_05x`, `A7fa_e226_05x`, `B6b_e197_05x` | as above | as above | 0.5× | s1, 3, 6, 10 | The same 2×2 at 0.5× |
| `B6b_e226_05x` | 6b16e298 | e226–e235 | 0.5× | s1, 3, 6, 10 | Reproduces your 0.5× chain (33.0 / 29.4 / 31.8) through our `stream` entry point |
| `B6b_e226_025x` | 6b16e298 | e226–e235 | 0.25× | s1, 3, 6, 10 | Reproduces your 0.25× chain (61.6 / 30.0 / 34.8) |
| `S6b_skip_025x` | 6b16e298 | e226–e228, then e232–e235 | 0.25× | s3, 4, 7 | **Do e229–e231 cause the collapse?** |
| `G6b_e226_1x` | 6b16e298 | e226–e235 | 1×, gate every step | every step | The deployment number: how many steps pass, and where it ends |

The gated arm accepts a step if z > 0 and the capped rate is under 20%. A rejected step rolls back both the bf16 weights and the fp32 master.

## Decision rules (written before any results)

1. **Does it reproduce?** `B6b_e226_05x` and `B6b_e226_025x` should collapse: s6 and s10 at least 20pp below the start. If they don't, `stream` and the wrapper differ, and we sort that out before reading anything else.
2. **Head or data?** Read the 1× cells; the 0.5× cells are a check.
   - **Data:** `A7fa_e226_1x` ends at least 10pp below its start at s10, and `B6b_e197_1x` ends at or above its start.
   - **Head:** the reverse.
   - **Both collapse, or both recover:** both factors contribute, or the outcome depends on step size. Read the 0.5× row.
3. **Skip test.** If `S6b_skip_025x` at s7 is at least 15pp above `B6b_e226_025x` at s10, the negative batches deepen the dip (reading 1). Otherwise, a deep dip is simply unrecoverable with stale data (reading 2).
4. **Gated arm.** Our prediction: 0–2 of 10 steps pass, and the end is within ±2pp of the start (safe, no gain).

## How to run

### Requirements
- **Training venv** (`PY`): torch 2.14.0, transformers 5.14.1, flash-linear-attention 0.6.0 and fla-core 0.6.0 (`--no-deps`), plus boto3, pynacl, numpy, safetensors and einops.
- **Eval venv** (`PYV`): vllm 0.32.0 and math-verify 0.9.0.
  - These need two separate venvs, because vLLM 0.32.0 wants transformers ≥ 5.16.1.
- **Source and models:**
  - the signed source tree (bundle `a2a03cb3…`);
  - the MiMo base checkpoint (the KL reference);
  - the checkpoint directories of heads 7fa0305a and 6b16e298.
- **Turn data** for e196–e206 and e226–e235. e196 is the fixed-text probe; it is not trained.

  ```bash
  python code/fetch_k1.py --out $DATA --epochs 196-206
  python code/fetch_k1.py --out $DATA --epochs 226-235
  ```
- **Replay host files.** `run_2x2.sh` downloads `/k1/replay/replay_k1.py` and `/k1/replay/host/*.py` from affine.io and applies `patch/support_renorm.diff`. The result is byte-identical to what our runs used. To use your own copy, set `HOST_RENORM`.
- **Hardware, per arm:**
  - one H200, shared by training and vLLM;
  - about 36 GB of pinned CPU RAM for the fp32 master (about 90 GB for the gated arm);
  - about 20 GB of disk for the candidate checkpoint.

### Command
```bash
PY=/path/venvA/bin/python PYV=/path/venvB/bin/python SRC=/path/affine_src DATA=/path/k1 BASE=/path/mimo-base \
CK_7FA=/path/7fa0305a CK_6B=/path/6b16e298 OUT=/path/results GPUS="0 1 2 3 4 5 6 7" bash code/run_2x2.sh
```

**What the script does:**
1. KL reference log-probs for the 20 trained turns, split across the GPUs.
2. Start evals of both heads on the gate set, about 18 min.
3. The 11 arms, spread round-robin over `GPUS`.

**Timing:**
- With fewer GPUs, the same arms simply queue.
- One training step takes about 1 min.
- One gate eval takes about 16 min at a 10% cap rate and about 40 min at 60%.
- The gated arm is the longest, at about 4 h.

**Gate set.** The default is `data/gate_B.json`: 977 tasks × 4 samples, T=1 / top-k 20 / top-p 0.95, `seed = 20261101 + task index`, graded by the completed-boxed rule in `code/gen_heldout.py`. None of its tasks were used in e196–e245. To use your own probe set, point `GATE` at a file in the same format (`index`, `prompt_ids`, `answer`). Exclude the tasks used in e197–e206 and e226–e235.

## What to send back

**Main outputs:**
- `OUT/summary.txt`: one line per scored step, each with z, diff, pass rate and capped rate against the start.
- `OUT/progress.txt`.
- `OUT/start_*/gate.jsonl.summary.json`.

**Per arm:**
- `OUT/<arm>/steps.jsonl`, which contains the per-step sinfo:
  - raw and published update norm;
  - tensors rel-capped;
  - realized / intended norm;
  - changed elements;
  - preclip gradient norm;
  - tokens outside the support.
- `OUT/<arm>/log.txt`.
- `OUT/<arm>/gate_s*.jsonl`, the per-sample gate results (small; lets us redo the paired stats).
