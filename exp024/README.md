# EXP-024: gate variance and local reproduction of the gate (e600, head 2cf21ce5)

The run was on 2026-11-04 on 3×H200, with the miner kit at affine@`eaf1ad20`, vLLM 0.32.0 for sampling, and torch 2.14 / transformers 5.14.1 for training. All z values here use **uniform π**, because the IRT π are private; the gate's own e600 grades score z 1.32 under uniform π.

## Runs (`results/`)

Every eval file has the gate's per-task format (`per_task`: n c w u b x tok L) plus a per-rollout list `rollouts` with entries `[block, task, j, length, stop, kind, how]`. Request seeds follow `sampler.request_seed(P, 'affine_gatepool', probe_index, j)`. P is the plan seed derived from the round seed: `prod` means the revealed seed, so the plan seeds equal your published `head_plan_seeds` / `plan_seed`; `sN` means `sha256('exp024:sN')`.

| file | model | draw | seeds |
|---|---|---|---|
| `H_Dp_prod.json`, `H_Dp_s1.json` | head 2cf21ce5, 12 rollouts per task (3 blocks × 4) | round 600 (479 tasks) | prod, s1 |
| `C1_Dp_prod.json`, `C1_Dp_s{1,2,3}.json` | gate candidate 2ba4137e, 8 per task | round 600 | prod, s1–s3 |
| `C2c_Dp_prod.json`, `C2c_Dp_s{1,2}.json` | local candidate: `train_step.py`, no `--master` (bf16 upcast), 1 GPU, `padded` | round 600 | prod, s1, s2 (paired with the C1 seeds) |
| `C0_Dp_s1.json` | head 2cf21ce5 scored as a candidate (null) | round 600 | s1 |
| `C4_D607_prod.json` | e600 file trained one step from 2ba4137e (local, upcast) | round 607 (491 tasks) | prod (607 cand seed) |
| `step_C2c.json`, `step_C4.json` | full `train_step.py` reports (incl. per-rollout trajectories) | | |
| `weights_C1_vs_C2c.json` | per-tensor diff, head → gate candidate vs head → local candidate | | |
| `summary.json` | output of `python3 summarize.py results` | | |

## Code (`stage/`)

- `evalrun.py` runs the gate's sampling and grading on a revealed round: the kit's `sampler`, `grade` and `baseline_v1` called unchanged, with the same request-seed rule.
- `train_ckpt.py` is `kit/train_step.py` run unchanged, except that every decoder layer of the trainable model gets activation checkpointing. Without it, the stock script runs out of memory on a 141 GB H200 for the e600 file (910 rollouts, mean 5.3k tokens, 46 capped). With it, one step takes about 23 minutes.
- `cmp_weights.py` is the per-tensor diff behind `weights_C1_vs_C2c.json`.
- `run2.sh` is the pod driver. Its `serve` function holds the vLLM command line we used, which needs `--limit-mm-per-prompt '{"image":0,"video":0}'`.
- `summarize.py` produces all the tables and needs only `stage/baseline_v1.py` (the gate module, verbatim) and `stage/data/round{600,607}.json` (the public reveals).

## Fingerprints of the local candidates

We did not keep the local candidate weights: the pod was deleted after the run. To check that a regenerated 1-GPU upcast candidate is the same one, compare against `step_*.json` and `weights_C1_vs_C2c.json`:

| | C2c (from 2cf21ce5) | C4 (from 2ba4137e) |
|---|---|---|
| objective_loss | 0.00011019664816558361 | 0.00011631613597273827 |
| preclip_gradient_norm | 0.9107210636138916 | 0.8926506638526917 |
| raw_update_norm | 0.022880917029732575 | 0.022880317220606506 |
| projected_changed_elements | 24,815,136 | 25,129,555 |

`weights_C1_vs_C2c.json` has, for each of the 760 tensors, the number of changed elements (`changed_b`) and the update norm (`nb`) of C2c. For example, `lm_head.weight` has 209,411 changed elements and norm 0.0018560029566287994.

## Commands

```sh
# training (kit env)
python3 train_ckpt.py --head HEAD --ref REF_87de8304 --batch batch600.json \
  --hotkey 5C4hsyrmfBHWf2RHRxR4BrjDzNaHvgW38sFyZS8ky5S5K2Zq --out CAND --device cuda --report step.json
# (move CAND/fp32-master.safetensors out of CAND before serving: there is no index file, so vLLM would load it too)

# serving (vLLM 0.32.0 env)
vllm serve CAND --served-model-name m --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching \
  --max-num-seqs 512 --limit-mm-per-prompt '{"image":0,"video":0}'

# eval
python3 evalrun.py --url http://127.0.0.1:8000 --tok HEAD --round round600.json --role cand --seed prod --epoch 600 --out out.json
```

`batch600.json` is the public `/k1/turns/600/batch.json` (sha256 d7ee274e…).
