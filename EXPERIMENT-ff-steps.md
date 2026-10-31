# Experiment: one step or a chain for a large free file?

**Question.** Miners now upload large free files: up to 125 prompts and 1,000 rollouts. The live trainer applies the whole file as one step. Our earlier evidence points the other way: split it into small sub-steps, chain them, and judge only the end. That evidence comes from the old 9-task, contract-sampled batches:
- six batches pooled into one step: −14.6pp;
- the same six as a 6-step chain: +2.4pp;
- 6b16e298 + e226–e235: 0/10 single steps passed the gate, while the 10-step chain ended +3.6pp (z +6.8).

This experiment asks the same question on real large free files, and separates the effect of chaining from the effect of the step recipe.

Everything runs with one command, `code/run_ff_steps.sh`. With 3 files on 8×H200, expect about 5–6 h.

## Arms

All arms in a file start from the same head and its published fp32 master.

| Arm | How the file is applied | Update | Scored at |
|---|---|---|---|
| `P_live` | Whole file as one step | Live: fp32 master + fresh-AdamW (≈ sign) step, no renorm | s1. Should land near the live candidate |
| `P_ours` | Whole file as one step | fp32 master + support renorm + gradient-direction step, 1× | s1 |
| `C_ours` | Chain: 9 prompts per sub-step in a seeded order, about 14 sub-steps for 125 prompts | Same as `P_ours` | ¼, ½, ¾ of the way and the end |
| `C_live` | The same chain | Same as `P_live` | ¼, ½, ¾ and the end |
| `C3_ours` (only with `EXTRA=1`) | Chain at 27 prompts per sub-step, about 5 sub-steps | Same as `P_ours` | ¼, ½, ¾ and the end |

**How the comparisons separate the effects:**
- `C_ours` vs `P_ours`: does chaining help with our step?
- `C_live` vs `P_live`: does chaining help with the live step?
- `C_ours` vs `C_live`: within a chain, how much does the step recipe matter?

**Which files.** Use at least 3 of the largest recent free files (ideally 100+ prompts each), from different miners, all trained (or queued) on the same head. RED files are fine: the question is whether they would have helped as a chain. Pass that head as `CK` and its published fp32 master as `MASTER`.

## Decision rules (written before any results)

1. **Sanity.** On most files, `P_live` is within ±3pp of the start with z < 1. That would be consistent with the live RED streak.
2. **Main question.** All comparisons are end vs. end on the same gate tasks.
   - **The chain wins** if, on at least 2 of 3 files, the end of `C_ours` beats both `P_ours` and `P_live` by at least 2pp with z ≥ 2.
   - **One step wins** if `P_ours` or `P_live` beats the end of `C_ours` with z ≥ 2 on at least 2 files.
   - Anything else is inconclusive.
3. **Our predictions:**
   - `C_ours` ends at least +1pp above the start on 2 of 3 files, possibly after a deep mid-chain dip.
   - `P_ours` ends at or below −5pp (the pooled gradient-direction step).
   - `C_live` ends at or below −5pp. The fp32 master with the sign step and no renorm drifts toward the length cap: 85.3% → 44.8% over 17 steps in EXP-023.
4. **Per file.** A file can be harmful however it is applied, for example if its rewards are wrong. Read the arms of each file together.

## How to run

The requirements are the same as `run_2x2.sh` (two venvs, signed source tree, MiMo base as the KL reference). On top of those:

| Variable | What it is |
|---|---|
| `CK` | The head checkpoint directory. Its tokenizer and `chat_template.jinja` render the files, exactly as `ff_format.py` does |
| `MASTER` | The head's published fp32 master (`*.safetensors`) |
| `FILES` | Paths to the free files, space-separated |

```bash
PY=/path/venvA/bin/python PYV=/path/venvB/bin/python SRC=/path/affine_src BASE=/path/mimo-base \
CK=/path/head MASTER=/path/head-fp32-master FILES="/path/f1.json /path/f2.json /path/f3.json" \
OUT=/path/results GPUS="0 1 2 3 4 5 6 7" bash code/run_ff_steps.sh
```

### How a file is handled
`bf16_exp.py --ff` reads it the way the live trainer does:
- **Cap:** whole prompts in file order, at 125 prompts and 1,000 rollouts.
- **Prompt:** `apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)`.
- **Output:** `encode(output, add_special_tokens=False)`, plus `<|im_end|>` when it is under 8,192 tokens. Exactly 8,192 tokens means capped.
- **Reward and advantage** follow v8.3.3h `_layout`: the miner's reward plus the soft overlong penalty, advantage over the group's own rollouts, and unresolved rows kept active.
- **Loss:** seq-mean-token-mean × 0.5635, gradient clip 1.0 per (sub-)step, and the step-recipe settings.

### What the script does
1. **Replay files.** It downloads your published replay files and builds the renorm host.
2. **KL reference pass.** It runs one per file and grouping.
3. **Start eval.** It scores the head on the gate set.
4. **Arms.** It runs the arms, one per GPU at a time.
5. **Summary.** It writes `summary.txt` with `compare_ff.py`.

### Built-in checks
- **fp32 master:** with `--start-master`, the run stops unless bf16(master) equals the head bit for bit. It also stops, listing the missing names, if the master's tensor names don't match the model parameters.
- **Sub-step log:** each sub-step logs its prompts, rollouts, order and `support_outside_tokens`. That last count shows how off-policy a file is; contract-sampled batches run at 0.002–0.07%.
- **Gate overlap:** gate tasks whose problem text appears in a file are left out of that file's comparisons and listed in `gate_exclude.json`. The match is a substring match after lowercasing and collapsing whitespace, at 40 characters or more. The live gate does not exclude them; we do, so that we measure generalization.

### Cost
- **Gradient work:** about the same for every arm. Each arm makes one pass over the whole file, about 2M tokens, whether it is split or not. That takes roughly 13–30 min.
- **Gate evals:** 16–40 min each. Chains have 4.
- **Per arm:** about 80 GB of host RAM and 20 GB of disk.

### Tested so far
I ran the code path on CPU with a synthetic free file built from public turns, both as one step and as a 9-prompt chain. That covered tokenization, rows, advantages, the micro-batch plan and gate-overlap detection with planted prompts. It has not run on a GPU in this form.

## What to send back
- `OUT/summary.txt`: every arm against the start, and the end-vs-end pairs.
- `OUT/progress.txt`.
- For each arm: `log.txt`, `steps.jsonl` (per-step update norms, changed elements, `support_outside_tokens`), `gate_s*.jsonl` and `gate_exclude.json`.
