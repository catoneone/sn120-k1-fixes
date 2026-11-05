# EXP-024 phase 5: why the miner kit's gradient differs from the gate's (e718, head 58bcc262)

Kit at affine@`9935e14`. One H200 per run. Activation checkpointing on (`--activation-checkpointing`) in every run.

## Code (`code/`)

- `train_live.py` runs `kit/train_step.py` unchanged, except that the trainer's own kernels are used:
  - `v83_fast_kernels.install()` is applied to the actor and to the KL reference;
  - `batched_log_probs` is routed to `fast_kernels.packed_log_probs`;
  - `engine='packed'`.

  This is one rank of `v83_dp` on a single GPU.
- `grad_probe.py` computes, for the same 24 rollouts of the e718 file, the gradient of L = Σ_i A_i · mean_t log p_t (full-vocab log p). The rollouts are 8 short, 8 mid and 8 long, up to 12,288 tokens. Gradients are accumulated in fp32 and saved per tensor. Modes:
  - `kit`: the kit's load_model with the native transformers forward;
  - `live`: + `v83_fast_kernels.install`;
  - `live_fusednorm`: `install(fused_norm=True)`;
  - `live_gdn_only`: only the GDN forward replaced; full attention stays on sdpa;
  - `kit_flaconv`: kit path with only the GDN short conv replaced by fla `causal_conv1d`;
  - `fp32ref`: fp32 weights, pure torch (`torch_chunk_gated_delta_rule`, `F.conv1d`, torch gated RMSNorm, sdpa).
- `cmp_grads.py` compares gradients pairwise, per tensor, aggregated by layer block × tensor class. `cmp_master.py` and `layer_cos.py` compare fp32-master updates (= per-tensor gradient directions under normgrad) per layer. `traj_diff.py` diffs the per-rollout trainer records of turn.json against a local report.

## Results (`results/`)

- `cmp_grads.json` and `cmp_grads_ablation.json`: the gradient comparisons.
- `m_gate_vs_L.json`, `m_gate_vs_K.json` and `m_L_vs_K.json`: master-update comparisons against the gate candidate 1b70bca8, where L = `train_live.py` and K = kit default.
- `step_L.json` and `step_K.json`: the trainer reports, with trajectories stripped.
- `probe_*.json`: per-rollout lengths, advantages and mean log p for each mode.
