# SN120 K=1 training: sampler mismatch, bf16 rounding, sign step

This bundle contains the findings, the arm code (EXP-022/023), the 750-task held-out set and the exact sampling settings. Start with `FINDINGS.md`. The grader issue (correct answers scored 0) is in `VERIFIER.md`.

| Path | What it is |
|---|---|
| `FINDINGS.md` | Problems found, the 17-step results table, recommended fix |
| `patch/support_renorm.diff` | Support renormalization (SC_MIS at lag 0) against the published `/k1/replay/host/` files (`v8211_batched.py`, `v83_fast_kernels.py`, `v83_dp.py`). Default off: `SUPPORT_RENORM = None` leaves the current loss unchanged |
| `code/bf16_exp.py` | All arms. Built on your `replay_k1.py`: same rows, KL reference pass, gradient pass, clip, fresh-AdamW step and per-tensor cap; only the last step changes |
| `code/gen_heldout.py` | Held-out generation and grading (vLLM) |
| `code/gate_eval.py` | Official-style gate eval used in the gated runs (1,000 tasks × 4) |
| `code/fetch_k1.py` | Downloads `/k1/turns/<e>/turn.json` and the batch documents (by sha256); epochs without a turn are skipped |
| `VERIFIER.md`, `code/lenient_grader.py` | The affine_math grader scores 481 correct finished answers as 0 in e196–e260. The script reproduces the live projection + `verify.py` (0 mismatches on 4,073 rollouts) and measures nine deterministic fixes |
| `code/gated.sh`, `code/summarize.py` | Gated-run driver; results summary |
| `EXPERIMENT-2x2.md`, `code/run_2x2.sh` | Head × batch 2×2 on 7fa0305a / 6b16e298, plus the extra 6b16e298 arms: one command, pre-registered decision rules |
| `EXPERIMENT-ff-steps.md`, `code/run_ff_steps.sh`, `code/compare_ff.py` | Large free files: one step vs. a chain of sub-steps, live step vs. our step; reads `affine-batch-v1` files the way v8.3.3h does |
| `data/heldout.json` | The 750 held-out tasks: `index`, `prompt_ids`, `answer`, `level`, `subject` |
| `data/gate.json` | The 1,000-task gate set used in the gated runs (same format) |
| `data/gate_B.json` | The 977 tasks of `gate.json` not used in any batch e196–e245 (live-head runs and the 2×2) |
| `results/verifier/regression.json` | Grader regression: label matrices before and after the fixes, counts per fix, every changed (task, fix) with the answers |
| `results/` | Per-arm step logs (`steps.jsonl`), fixed-text probes (`snapshots.jsonl`), single-step results (`stage1/`), and per-sample generation results (`gen_*/trunc.jsonl`, `full.jsonl`, `summary.json`) |

## Held-out set

- **Pool:** `original-math7496.tasks.json` (branch `validator/v829i-promotion-gate-20261026`, `assets/`).
- **Held-out:** the 750 indices of that pool that are *not* in the mining environment (`manifest-e000203.json` → `environments[0].indices`, 6,746 tasks). None of them appear in the K=1 batches e196–e213.
- **Prompt:** built from the task's `data` and encoded with the MiMo tokenizer (`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` at rev `2367e865…`), without adding special tokens. The model generates `<think>` itself.

  ```
  <|im_start|>system\n{system_prompt}<|im_end|><|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n
  ```
- **Gate set (gated runs only):** 1,000 indices drawn at random (seed 20261029) from the mining pool. It excludes every task used in e196–e213 and the 750 held-out tasks.

## Exact sampling settings (`gen_heldout.py`)

- **Engine:** vLLM 0.32.0, `dtype=bfloat16`, `max_model_len=12288`, `enable_prefix_caching=True`, `seed=0`.
- **Truncated arm:** `temperature=1.0, top_k=20, top_p=0.95`. This is the contract sampling.
- **Full-vocab arm:** `temperature=1.0, top_k=-1, top_p=1.0`.
- **Per task:** `max_tokens=8192`, `stop_token_ids=[248046, 248044]`, `skip_special_tokens=False`, `n=k` (k = 2 for most arms, 1 for the round-2 arms), `seed = 20261029 + task_index`. Using the same seed per task gives common random numbers across checkpoints, so the comparisons in `FINDINGS.md` are paired on sample j=0.
- **Gate eval (`gate_eval.py`):** truncated arm, `n=4`, `seed = 20261101 + task_index`. Accept if paired z > 0 (per-task mean difference divided by its standard error) and the candidate's capped rate is below 20%.
- **Grading** (close to the official completed-boxed rule): the sample counts as correct only if
  - `finish_reason == 'stop'` (EOS),
  - `</think>` is present,
  - and the last `\boxed{}` after `</think>` passes `math_verify==0.9.0` `verify(parse('$'+answer+'$'), parse('$'+box+'$'))`.

## Training arms (`bf16_exp.py stream`)

- **Common setup:** start checkpoint 7fa0305a; K=1 batches e197–e213 applied in order, every step accepted (the gated runs are the exception). The probe text is the e196 batch, which is not used for training.
- **`--turns`** takes a range (`197-213`), a reversed range (`210-197`) or a comma list (`226-228,232-235`).
- **`--ff FILE`** trains on a free file instead of `--turns`: `--ff-per-step 0` applies it as one step, `--ff-per-step 9` as a chain of 9-prompt sub-steps (`--ff-seed` fixes the order). `--start-master DIR` starts from a published fp32 master.
- **Environment:** torch 2.14 / transformers 5.14.1 / fla 0.6.0, with your signed source tree (bundle `a2a03cb3…`) and your `/k1/replay/host/` files. The `fp32_renorm` arms use the host files with `patch/support_renorm.diff` applied.

| Arm | Flags |
|---|---|
| live recipe | `--arm rtn` |
| fp32 master | `--arm fp32` |
| fp32 + renorm | `--arm fp32_renorm --host <patched host>` |
| stochastic rounding | `--arm sr` |
| random-sign control | `--arm randsign` |
| fp32 + renorm + gradient-direction step | `--arm fp32_renorm --opt normgrad` |
| same, smaller or larger step | add `--step-mult 0.5` (scales each tensor's step) |
| fp32 + renorm, positive advantages only | `--arm fp32_renorm --adv-pos` |
| gated | add `--gate-tasks data/gate.json --cand-dir DIR` (a rejected step rolls back the bf16 weights and the fp32 master) |

`--opt normgrad` keeps each tensor's step norm equal to the official capped sign step and changes the direction from sign(g) to g: Δ_t = ‖Δ_sign,t‖ · g_t / ‖g_t‖.

## Status

- The ungated results are final.
- The gated runs (round 3) and the ungated gate-set trajectory (round 4) are finished; see the Update section of `FINDINGS.md`. Their logs are in `results/round3/`.
