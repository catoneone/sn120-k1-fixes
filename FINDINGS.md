# What we found in the MiMo K=1 training, and a recipe that trains (2026-10-29)

Code, data and exact settings are in this repo; see `README.md`.

---

We spent some GPU time replaying your published K=1 steps with your own `replay_k1.py`. Our replay matches yours: the objective is bit-identical and 41,068,781 elements change against 41,068,811 published. We found three problems that stack on top of each other, and a combination of fixes that makes the model improve instead of drift.

**Setup.** We start from 7fa0305a and apply all 17 public K=1 batches (e197–e213) in order, accepting every step. The loss, clipping and caps are yours. We then evaluate on 750 held-out math tasks that do not overlap the batches. Generation uses T=1, top-k 20, top-p 0.95 and 8,192 max tokens, and we also run a full-vocab pass. Pass rates are paired on the same sampling seeds.

## Problems

1. **The sampler and the gradient do not match.** Rollouts are sampled from the truncated distribution (top-k 20, then top-p 0.95), but the policy gradient is taken on full-vocab log p.
   - On the public batches, almost the entire update comes from negative advantages on capped rollouts (cos 0.995 with the total update).
   - That term is almost exactly the negative of the truncation correction (cos −0.99). It mostly lowers the probability mass of the sampler's support and pushes it into tokens the sampler never draws.
   - Under full-vocab sampling the trained head ends inside `<think>` with EOS in 16.5% of answers; the base model does this in 0.2%.
   - Miners and the gate both sample truncated, so they never see this tail and never penalize it.

2. **The bf16 rounding in K=1 turns each step into almost nothing.** The fresh-AdamW step is lr·sign(g), with lr 5e-7 and a 2e-5 per-tensor relative cap, rounded to nearest bf16 with no fp32 master.
   - Only weights with |w| < 2^-12 can move. That is 0.46% of elements and 10% of the update norm.
   - Only 0.8% of the step's first-order effect on the batch loss and on held-out log-likelihoods survives.
   - The real effect of one step is smaller than the numerical noise of a bf16 forward pass, so the gate is mostly comparing noise.

3. **The sign step itself is harmful once updates really apply.** sign(g) moves every coordinate by the full step, so small but consistent gradient components get amplified to the size of the main signal.
   - A control with the same per-step size but random signs changes nothing: 86.9% vs 84.9% at the start.
   - So the damage comes from the amplified direction, not from the step size or noise.

4. **The bf16 rounding is currently hiding problems 1 and 3.** If you fix only bf16, the model collapses within 17 steps. Even with today's rounding, accepting all 17 steps costs 8.4pp. In the live run, the gate promoted only 2 of these 17 turns.

## Results (17 steps from 7fa0305a; truncated sampling unless noted)

| Recipe | Pass | vs start (95% CI) | Capped | Full-vocab: EOS, no answer |
|---|---|---|---|---|
| Start (7fa0305a) | 84.9% | — | 12.5% | 21.7% |
| Random-sign control (same step size) | 86.9% | +2.0 (−0.1, +4.3) | 11.5% | 22.1% |
| Live recipe (bf16 round-to-nearest) | 76.5% | −8.4 (−11.1, −5.7) | 21.5% | 38.0% |
| fp32 master only | 45.2% | −39.7 (−43.5, −36.4) | 53.7% | 69.5% |
| fp32 master + support renormalization | 66.4% | −18.5 (−21.5, −15.3) | 30.9% | 51.3% |
| fp32 master + renormalization, positive advantages only | 85.9% | +0.9 (−1.6, +3.2) | 11.7% | 26.0% |
| Stochastic rounding | 9.5% | −75.5 | 82.3% | 97.9% |
| **fp32 master + renormalization + gradient-direction step** | **90.5%** | **+5.6 (+3.5, +8.0)** | **7.5%** | **0.1%** |

Under the last recipe, the full-vocab and truncated pass rates are now the same (90.4% vs 90.5%), and the EOS-mid-think tail is gone.

## What we suggest (all three together)

1. **Support renormalization in the policy-gradient term.** This is your SC_MIS at lag 0.
   - Use log q, renormalized over the sampler support that is replayed from the trainer's own logits.
   - Keep KL and the derail mask on full-vocab log p.
   - Drop the roughly 0.03% of tokens that fall outside the support from the PG term.
   - We have a default-off patch against the published `v8211_batched.py`, `v83_fast_kernels.py` and `v83_dp.py`. The packed 8-GPU path needs it too.
2. **fp32 master weights kept across turns.** Publish bf16 = round(master), plus a bf16 residual so that turns stay replayable.
3. **Replace the sign step with a gradient-direction step.** Per tensor, use Δ_t = ‖Δ_sign,t‖ · g_t / ‖g_t‖, which keeps today's per-tensor step size and caps but follows g instead of sign(g). We have not tested a persistent-state Adam. Adam also normalizes each coordinate, so a small gradient that is consistent across batches can still get a full-size step, which is exactly the component that hurts here. Please do not assume Adam is equivalent.

Two things to avoid: stochastic rounding, and fixing bf16 without fixes 1 and 3.

## Caveats

- One start checkpoint, one seed, 17 steps, on the public batches. Those batches were sampled from the live heads, so they are slightly off-policy for the arms that move further.
- The last two arms in the table were evaluated with 750 × 1 samples; the others with 750 × 2.
- Next, we want to replicate from 41a1afd0 and 44fb50bb, run longer, and test a persistent-state Adam as a separate arm.

We are happy to share the code (based on `replay_k1.py`) and all raw results.

## Update: with an official-style gate, and the trajectory on the gate set (2026-10-29)

**Gated runs.** We reran the two fp32 + renormalization arms with a gate on every step.
- Gate: 1,000 private tasks × 4 samples, common random numbers, accept if z > 0 and capped < 20%. A rejected step rolls back both the bf16 weights and the fp32 master. The gate set does not overlap the batches or the held-out set, and its noise is about 1pp, the same as the live gate.
- Fourteen steps, e197–e210. The table shows each candidate minus the current head on the gate set, in pp; A = accepted.

| Batch | 197 | 198 | 199 | 200 | 201 | 202 | 203 | 204 | 205 | 206 | 207 | 208 | 209 | 210 | Accepted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gradient-direction step | −1.3 | −5.6 | −7.0 | −2.5 | −3.8 | −8.0 | −2.2 | −0.1 | −2.0 | −8.9 | **+1.9 A** | −1.5 | +0.5 A | +0.9 A | 3/14 |
| Sign step | 0.0 A | −0.3 | −0.9 | −0.9 | −0.2 | −0.4 | −0.3 | +0.1 A | −0.6 | −0.2 | +0.1 A | −0.7 | −0.4 | +0.2 A | 4/14 |

The first ten gradient-direction candidates were rejected, so e197–e207 are all single steps from the same start.

**Ungated trajectory, scored on the gate set.** Same recipe, 17 steps, every step accepted:

| Step | 0 | 1 | 3 | 6 | 10 | 17 |
|---|---|---|---|---|---|---|
| Gate-set pass | 85.9% | 83.7% | **68.1%** (capped 29.9%) | 88.3% | 89.4% | **90.0%** |

**Held-out results** (750 tasks, paired on j=0 against the start, task-bootstrap 95% CI):

| Recipe | Pass | vs start | Capped | Full-vocab: EOS, no answer |
|---|---|---|---|---|
| Start (7fa0305a) | 84.9% | — | 12.5% | 21.7% |
| Random-sign control (noise floor) | 86.9% | +2.0 (−0.1, +4.3) | 11.5% | 22.1% |
| Gated, sign step (4/14 accepted) | 87.1% | +2.1 (0.0, +4.1) | 10.8% | 22.4% |
| **Gated, gradient-direction step (3/14 accepted)** | **88.7%** | **+3.7 (+1.5, +6.1)** | 8.1% | 12.4% |
| Ungated, gradient-direction step (17 steps, rerun) | 88.9% | +4.0 (+2.0, +6.3) | 8.5% | 0.1% |

**What this shows**

1. **Single steps are not a reliable measure of batch quality.**
   - From the same start, 10 of 11 batches give a negative single step; only e207 is positive.
   - In the ungated run, the model drops to 68% at step 3 and is back at 88% by step 6. The batches that bring it back (e200–e202) each hurt when applied alone from the start.
   - So a step's effect depends on the current state: the gradient-direction step overshoots, and later steps correct it.
2. **A per-step gate works well with the gradient-direction step.**
   - Only 3 of 14 steps pass, yet held-out improves by +3.7pp, close to the ungated +4.0pp.
   - It never goes through the dip. The 68% state would have been a visible collapse if published.
   - The cost: 11 of 14 batches are discarded, and only half of the full-vocab EOS tail is removed.
3. **A gate plus the sign step does not train.** Every single step is within ±1pp, and the held-out change (+2.1pp) matches the random-sign noise floor.
4. **The ungated recipe reproduces.** The rerun gives +4.0pp held-out (first run +5.6pp) and +4.1pp on the gate set.

**Recommended setup, updated:** fp32 master kept across turns, support renormalization in the PG term, gradient-direction step, and the gate kept as a safety net. To waste fewer batches, the next things to try are a smaller step (less overshoot) or gating every few steps instead of every step.

Raw logs for these runs are in `results/round3/`.

**Reversed batch order** (same gated recipe, e210 → e197):

| | Accepted batches | Gate-set end | Held-out vs start (95% CI) | Full-vocab EOS, no answer |
|---|---|---|---|---|
| Forward | e207, e209, e210 (3/14) | 89.2% | +3.7pp (+1.5, +6.1) | 12.4% |
| Reversed | e207, e205, e203, e200, e199 (5/14) | 89.7% | +5.5pp (+3.2, +8.0) | 0.3% |

- **Only e207 is accepted in both orders.** Its single step from the start is +1.9pp forward and +2.1pp reversed, the only batch that is reliably positive.
- **The other batches flip sign depending on the state.**
  - e209 and e210 hurt from the start (−2.5 and −5.4pp) and help only after e207.
  - e205, e203, e200 and e199 hurt from the start (−2.0, −2.2, −2.5 and −7.0pp) but are positive or near zero after e207.
- **Both orders end up at about the same place.** So whether a batch counts as "good" depends mostly on the state it lands on, not on the batch itself.
- **Paying miners for single-step gate wins therefore rewards queue position more than data quality.** A per-miner turn that applies a larger batch as several sub-steps, and is judged only at its end, would give a cleaner signal.

**One big step vs. a chain.** We also tested applying the six batches e197–e202 as a single step from the start: each batch's clipped gradient computed at the start, averaged, then one gradient-direction step. Scored on the gate set against the start (85.9%):

| Update | Gate-set pass | vs start |
|---|---|---|
| Six batches as a chain of 6 steps (ungated) | 88.3% | +2.4pp |
| Six batches pooled into one step, 1× step | 71.3% | −14.6pp (z −18.4) |
| Six batches pooled into one step, 4× step | 60.2% | −25.7pp |
| Each batch alone, one step from the start | — | −1.3 to −8.0pp |

**What this means**
- **The pooled step is worse than any single batch.** Averaging cancels the batch-specific noise and keeps the shared direction, and from the start that shared direction pushes toward the length cap.
- **The gain comes only from chaining.** After each step the state changes, and the next gradient corrects the overshoot.
- **For a mechanism with larger per-miner batches:** apply the batch as a chain of sub-steps and judge the turn at its end, not as one big step.

**From the live head 1636928e, and two step variants** (ungated, 6 steps on e240–e245, which were sampled from this head; gate set of 977 tasks × 4, start 84.1%):

| Step | Gradient-direction | Per-row normalized + capped embed / lm_head | Per-element clip at 4× the sign step |
|---|---|---|---|
| 1 (e240) | 68.7% (−15.4) | 68.8% (−15.3) | **84.0% (−0.1)** |
| 3 (e242) | **88.2% (+4.1)** | **88.4% (+4.3)** | 84.2% (+0.05) |
| 6 (e245) | **87.8% (+3.7)** | **88.1% (+4.0)** | 82.2% (−1.9) |

- **The gradient-direction step does not collapse from this head.** It dips and then recovers to +4pp by step 3, as it did from 7fa0305a, so it does not reproduce the 6b16e298 / e226–e235 collapse.
  - In the public turn files, the batches that came right after the dip in that chain (e229–e231) push toward *longer* answers. The batches after the dip in our chains push toward finishing.
  - Stale replayed batches cannot react to the dip; on-policy batches, or a gate, would.
- **Per-row capping changes nothing.** The dip does not come from the high-frequency formatting rows of embed or lm_head, so my earlier guess was wrong.
- **Per-element clipping removes the dip and the gain together.** It ends slightly negative, much like a sign step: the few large-moving elements cause both the overshoot and the later improvement.
- **We found no simple step-size fix that avoids the overshoot and still learns.** The safest options remain the gradient-direction step with a gate, or a per-miner chain of sub-steps judged at its end.
