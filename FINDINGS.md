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

## Status: gated runs in progress (2026-10-29 15:45Z)

We are now rerunning the two fp32 + renormalization arms with an official-style gate on every step: 1,000 private tasks × 4 samples, common random numbers, accept if z > 0 and capped < 20%, and roll back the fp32 master on rejection. Early single-step results from 7fa0305a, on the gate set:

| Batch | Gradient-direction step | Sign step |
|---|---|---|
| e197 | −1.3pp (z −2.7) RED | +0.03pp (z +0.06) GREEN |
| e198 | −5.6pp (z −10.6) RED | −0.27pp (z −0.6) RED |
| e199 | −7.0pp (z −11.5) RED | −0.85pp (z −1.95) RED |

So single gradient-direction steps from this start are strongly negative, and their ranking across batches matches the sign arm's. That conflicts with the +5.6pp after 17 ungated steps, and the two possible reasons are not yet separated:
- (a) batch quality varies and the later batches are the good ones;
- (b) single gradient-direction steps overshoot, and the gain only appears when steps accumulate. In that case a per-step gate blocks the path.

The remaining gated steps will tell (a) from (b), and we will update this file. Until then, treat the "gradient-direction step" recommendation as provisional when it is combined with a per-step gate.
