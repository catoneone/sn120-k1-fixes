**Proposal: four changes that loosen v2 training limits**

Short version: with a fresh Adam per file and 1 to 4 full-batch steps, every v2 update is a sign step of fixed size. The data picks the sign of each weight and nothing else. The LoRA also cannot reach 24 of the 32 attention layers, and the file caps keep files near 500 rollouts. We propose four changes. None of them adds a new validator rule; each gives miners more room, and the gate still decides what is applied.

- **A. `minibatches`** (miner, 1 to 8000, default 1): more Adam updates inside one file for the same tokens.
- **B. `global_norm_cap`** (miner, 0.01 to 6, default 6): a direct step-size setting, with a ceiling that keeps today's range.
- **C. Linear-attention projections** as optional `target_modules`: `in_proj_qkv`, `in_proj_z`, `out_proj`.
- **D. Higher file caps**: `max_train_token_steps` 8M → 16M, `max_prompts` 250 → 2000, `max_rollouts` 2000 → 8000, `max_batch_bytes` 64 → 128 MiB.

With the default settings the trainer gives the same LoRA as today, bit for bit. A patch and a CPU test are in `v2/patch/` and `v2/code/`. One check should run on a GPU before C goes live (section 6).

---

## 1. What is wrong now

**The code.** `gpu/train_job.py:315` builds a new `torch.optim.Adam` for each file (betas 0.9/0.999). On the first step, after bias correction, m̂ = g and v̂ = g², so each weight moves by lr · g / (|g| + eps) ≈ **lr · sign(g)**. Two side effects:
- `max_grad_norm` does nothing on step 1. The clip scales every element by the same factor, and that factor cancels in g / |g|. llms.txt says it is the step-size cap; in the code it is not.
- B starts at zero, so the gradient of A is exactly zero on step 1. With `steps = 1` (the default, and what most winners use) A never trains.

**The round data (r001 to r027).** Every entry with lr 8e-5 / rank 64 / alpha 128 has `step_norm` 0.72 to 0.75. Every default entry has 0.044 to 0.046. The files range from 138 to 752 rollouts. A fit gives step_norm ≈ 570 · lr · alpha / √rank. File size does not appear in it.

**A controlled test (official trainer, the r000027 winner file, current head):**

| Arm | Data | Gradient norm | step_norm |
|---|---|---|---|
| T1 | the file as published, 240 rollouts | 0.038 | 0.735 |
| T2 | first half of the file | 0.044 | 0.738 |
| T3 | rewards swapped inside each pair (pure noise) | 0.026 | 0.732 |

The noise file has a 30% smaller gradient, a direction nearly orthogonal to the real one (cos 0.003), and the same step. T1 matches the 0.732 that r000027 reported, so our local run follows the live path.

**More steps of a fresh Adam do not fix it.**
- We cut the same file into 8 mini-batches, one Adam update each. The step was about 2.4 for the original, the half and the noise file alike: still independent of the data. Eight updates of a fresh Adam move like a random walk (√8 ≈ 2.8).
- r017 used `steps = 2` and reported step_norm 1.08, which is 1.47 times the one-step value. A simulation of Adam on two gradients with per-element correlation c puts 1.47 at c ≈ −0.2 to 0. So on the same data, the second step's gradient was nearly unrelated to the first. `steps` acts as a step-size multiplier of about √k to k. It adds compute, not information.

**Why it matters.** The only setting that changes a file's effect, apart from its content, is the step size. From r014 on, more than 90% of entries use the same 0.73 recipe. From r022 the mean entry z went +0.91, −0.05, −0.93, with the null pass rate flat. A shared step size pointed at noise fits this. We have not proven it is the cause.

## 2. Change A: `minibatches` (miner-settable, 1 to 8000, default 1)

**What it does.** `steps` keeps its meaning: passes over the whole file. Each pass shuffles the rows (seeded by the job seed, so a replay gives the same LoRA) and cuts them into `minibatches` groups. Each group gets one Adam update, and the Adam state carries over. A file with `steps = 1, minibatches = 16` gets 16 updates, each on a fresh gradient at the current weights, for the same tokens as one update today.

**q_old.** With mini-batches, a row may first train after other updates. So q_old, the mask and the token count of every row are scored on the head, in one forward pass with no graph, before the first update. Without this the PPO ratio of later mini-batches would be measured from an already-moved policy.

**Cost.** Tokens per pass do not change, so the admission rule `steps × tokens ≤ max_train_token_steps` stays as it is. The q_old pass adds about one forward per row, roughly 25% of a training pass with the current gradient checkpointing; this is our estimate from the code, not a measurement. We kept the admission formula unchanged to keep the rule simple; the extra pass costs every miner the same, and section 5 shows it fits the time budget.

**Per-row weighting.** With one mini-batch the loss divisor stays `len(rows)`, as today. With more, it is the number of rows in the mini-batch.

**Why the bound is 8000 (= `max_rollouts`), in effect no limit.** At the top a mini-batch is a single rollout; empty mini-batches are skipped. A tighter bound protects nothing:
- The step size is bounded by `global_norm_cap` (change B), however many updates there are.
- Compute does not grow. Every rollout still goes through the model once per pass, and `train_rows` already runs one row at a time, so small mini-batches are not less efficient per token.
- What does grow is the fixed cost of each update: the Adam step, summing gradients across replicas and copying weights back. For about 120M LoRA parameters (rank 64) we estimate about 0.1 s per update; we have not measured it. Even 8000 updates would add only minutes.
- A file that is cut too finely and runs past the training time budget fails on its own GPUs; the other files of the round train on theirs. The cost falls on the miner who chose it, unlike `steps`, which multiplies compute and has to stay under the admission cap. A mini-batch with fewer rows than the file's GPUs leaves some GPUs idle, but with 6 files on 8 GPUs a file usually gets one GPU.

We suggest one line in llms.txt: each update has a fixed cost of about 0.1 s that the admission cap does not count.

## 3. Change B: `global_norm_cap` (miner-settable, 0.01 to 6, default 6)

**What it is.** It already exists: `save()` computes the norm of the whole update, (alpha/rank) · √(Σ ‖B·A‖²_F), and if it is above `global_norm_cap` it scales every `lora_B` by cap / norm. The update shrinks; its direction stays. `step_norm` is the norm after the cap, `step_norm_before_cap` the norm before it. Today the cap is null, so it never acts.

**Why a miner should set it.** It is the direct way to choose how far the update goes. Today a miner has to get there through lr, alpha, rank and steps. With mini-batches that gets harder, because the norm after many updates is hard to predict from lr.

**Why a ceiling of 6.** The current bounds allow about 6 at most (lr 1e-4, alpha 128, rank 8, steps 4, with the step-to-step correlation seen in r017; up to about 10 if the steps were fully aligned). The largest step seen in a round is 2.18 (r002). With mini-batches a file can make thousands of updates; even 4 steps × 32 mini-batches, 128 updates moving roughly like a random walk, could reach about 30. A default and ceiling of 6 keep the reachable range where it is today. No current recipe comes close to 6, so the default changes no current LoRA.

## 4. Change C: linear-attention projections as optional `target_modules`

**The gap.** The head has 24 linear-attention (Gated DeltaNet) layers and 8 full-attention layers. `target_modules` may only name `q/k/v/o_proj` (the 8 full-attention layers) and the MLP projections. No LoRA can touch the attention of the other 24 layers: three quarters of the model's attention cannot learn.

**The change.** Add `in_proj_qkv`, `in_proj_z` and `out_proj` to the choices of the `target_modules` bound. The default recipe keeps the 7 current names, so nothing changes unless a miner opts in. We left out `in_proj_a` and `in_proj_b` (4096 → 32 each, the decay and beta gates): they are tiny, and vLLM packs them into one fused layer, which is one more thing to check.

**What we checked in the code.**
- Training: in transformers' Qwen3.5, the three are plain `nn.Linear` (`modeling_qwen3_5.py:536-541`), so `add_lora` wraps them by name like the others. No other module of the model (including the vision tower) has these names.
- Serving: vLLM 0.32 supports LoRA on these layers for Qwen3.5. `Qwen3_5ForConditionalGeneration.packed_modules_mapping` maps `in_proj_qkv` and `in_proj_z` onto the fused `in_proj_qkvz`, and the GDN layer builds the two projections separately when LoRA is enabled (`layers/mamba/gdn/qwen_gdn_linear_attn.py`). `out_proj` is a `RowParallelLinear`.
- Fold: `weight + scale · B @ A` is the same for any linear layer. The new shapes (8192 × 4096 for `in_proj_qkv`, 4096 × 4096 for the other two) are already in `fold_kernel.SHAPES` as `q` and `o`, so the arithmetic self-check covers them.

**Cost.** The LoRA gets about 40% more parameters at any rank, and the fold touches 72 more weight matrices. Training compute is almost unchanged: the base forward and backward dominate.

## 5. Change D: higher file caps

**Which cap binds today.** Public winner files have 6K to 12K tokens per rollout. At 8M token-steps, `steps = 1` allows only about 700 to 1,300 rollouts, well below `max_rollouts` = 2000. Most winners use 2 rollouts per prompt, so `max_prompts` = 250 stops them at 500. So the real caps are the token-step limit and the prompt count.

**The change.**

| Param | Now | Proposed | Why |
|---|---|---|---|
| `max_train_token_steps` | 8M | 16M | the main cap; fits the time budget (below) |
| `max_prompts` | 250 | 2000 | should not bind before the token cap |
| `max_rollouts` | 2000 | 8000 | the same, for files with short rollouts |
| `max_batch_bytes` | 64 MiB | 128 MiB | 16M tokens of text is about 50 to 60 MB |

`steps` stays at 1 to 4.

**Training time.** A linear fit to the 14 public winners (wall time against estimated token-steps; tokens estimated as characters / 3) gives about 160 s fixed plus about 14K token-steps/s. That predicts about 12 minutes at 8M and 21 minutes at 16M. Adding the 25% q_old pass of mini-batch files gives about 26 minutes, inside the 40-minute training budget. The fit is rough: the token counts are estimates, and the GPUs per file vary with the number of files in a round (`n_gpus // n_batches`). Measuring one 16M file on the box would settle it.

**Value.** Our earlier measurement on a v1 head found that larger files barely helped (independent 1K-rollout batches had gradient cos ≈ 0.003). The v2 head restarted from genesis, so this may differ, and the cost of trying is only training time. The gate shows whether bigger files win.

## 6. Patch and tests

Files: `v2/patch/v2-trainer-relax.diff` (against `params.py` and `gpu/train_job.py` on the v2 branch; both files are unchanged on the branch as of 2026-11-17) and `v2/code/test_v2_trainer_relax.py` (setup steps in its docstring). Apply from `v2/` of a checkout with `patch -p2 < v2-trainer-relax.diff`, or from the repo root with `patch -p1`.

- `params.py`: `Recipe.minibatches = 1`; `Recipe.global_norm_cap = 6.0` (was None); `MinerBound` entries for `minibatches` (1 to 8000, integer) and `global_norm_cap` (0.01 to 6); `LINEAR_ATTENTION_PROJECTIONS` added to the `target_modules` choices; the four new caps as defaults; checks in `__post_init__`.
- `gpu/train_job.py`: `Replica.score_old()` (the first-pass scoring of `train_rows`, without a gradient); the training loop with mini-batches; a row moves its q_old and mask to whichever replica trains it; `minibatches` in `recipe_values`; `forward_tokens` counts the extra pass. With `minibatches = 1` the loop runs exactly the old code path.
- `admission.py`, `trainer.py` and the fold are unchanged. `trainer._check` already enforces `global_norm_cap` on the report.
- The live values in mining.json need the same changes: `recipe.global_norm_cap` 6, the four caps, and the new bounds.

On CPU, with a tiny random Qwen3 model and the official code, we checked:
- `minibatches = 1`: the LoRA is bit-for-bit equal to the official trainer's (steps 2).
- `steps = 2, minibatches = 4`: 8 updates; `minibatches = 8000` on 24 rows: 48 updates (one per row per pass).
- a miner cap of 0.05 cuts a 0.52 update to exactly 0.05.
- `minibatches` 0, 2.5 and 8001 and caps 0.001 and 7 are rejected with the usual bound message.
- `target_modules` with `in_proj_qkv`, `in_proj_z`, `out_proj` is accepted; `in_proj_a` is rejected.
- the new caps pass the `Params` checks.

Not tested: anything on GPU with the real head, the multi-GPU path where rows move between replicas, a LoRA on the linear-attention layers end to end, and any effect on gate scores.

**One check before C goes live.** If vLLM applied a linear-attention LoRA differently from the trainer, the gate would score one model and the fold would apply another, and the gate could not catch it. A cheap test: train one small LoRA with the three new modules, then compare per-token log-probabilities of a few fixed rollouts between the HF trainer with the adapter and vLLM with the adapter, next to the same comparison for an MLP-only LoRA. One GPU, about half an hour. We can run it, or you can run it on a box.

Also: llms.txt line 82 should drop the claim that `max_grad_norm` caps the step size.

## 7. What this does not fix

None of these changes makes the step size follow the signal in the data. In the 8 mini-batch test above, the noise file still moved as far as the real one. That needs Adam state kept across rounds (a long-run second moment used to normalize each file's gradient) or an optimizer without per-weight normalization, such as SGD. Both are larger changes, so we left them out of this proposal.
