# Task 0 — Solution summary
### Team `the_silent_subcarriers`

## Problem
For each of the 8 public conditions {Dipole, Log} × position {1, 2, 3, 5} we are given
8,000 training pairs (16×16 binary RIS configuration → measured complex CSI, 242
sub-carriers = 484 real numbers) and must predict the CSI for 2,000 unseen
configurations. We train one independent predictor per condition (8 in total).

**Metric.** The official Task-0 score is the normalized MSE (normalized squared
Frobenius error) `‖Ĥ−H‖²_F / ‖H‖²_F`, averaged over the 8 conditions. Lower is better.

## Key observations
1. **The channel is nearly affine in the RIS bits, plus local coupling.** Turning an
   element ON/OFF changes its reflection (ON ≈ +1, OFF ≈ −0.4); the dominant
   configuration dependence is linear in those reflections plus *pairwise* terms
   `s_i·s_j`. Long-range coupling is unidentifiable: there are C(256,2)=32,640 possible
   pairs but only 8,000 training configs, so only **near-neighbour** pairs can be
   estimated. We keep the 1,808 closest pairs (Manhattan distance on the panel).
2. **The output is low-rank.** Across configs the 484-dim CSI lives in a ~rank-9–15
   subspace (Log) / ~rank-21–25 (Dipole). We decode every model through a fixed
   **rank-32 SVD basis** of the training channel, which both regularises and shrinks the
   parameter count.
3. **The supervised targets are already denoised** — `train.csv` gives the per-config
   frame-averaged channel — so there is no separate denoising lever to pull.

## Model (one independent predictor per condition)
```
prediction = convex_blend_per_component( quad , cnn_ensemble )
             then: replace structured configs with the macro-block model
```
* **quad** — closed-form ridge regression of the rank-32 SVD coefficients on
  `[bias, 256 reflections, 1808 near-neighbour products]` (λ=100), solved as a single
  linear system. Captures the affine + local-coupling physics. ~82k params.
* **cnn_ensemble** — small 3×3 convolutional networks that pick up the residual
  non-linear structure the quad misses. Each net: a learnable per-element complex
  reflection "lift" → conv stem → residual conv blocks with squeeze-excite → (Log only)
  one self-attention layer over the 256 panel positions (global coupling a conv cannot
  see) → MLP, decoded through the same SVD basis. **Log: 4 seeds + attention; Dipole:
  2 seeds, no attention** (Dipole is easier and needs less capacity). AdamW + cosine
  schedule, early-stopped on a train-internal split.
* **convex blend** — for each SVD component k, `a_k·quad_k + (1−a_k)·cnn_k` with weights
  `a_k ∈ [0,1]` fit on a **held-out training out-of-fold split** (never on the evaluation
  configs). Blending helps because the quad and the CNN make different errors per
  frequency mode, which a single global weight cannot exploit.
* **structure refinement** — the public config set contains highly structured
  "macro-block" layouts (config_id 1–16 are four uniform 8×8 blocks; config_id 17–528 are
  nine uniform blocks on a 5/5/6 grid). For those test configs the channel is an exact
  affine/quadratic function of just 4 or 9 macro-bits, so we overwrite the blended
  prediction there with a tiny closed-form macro-block model fit on the **training**
  configs of the same family (104 such configs per condition are refined).

## What we found (and what does *not* help)
This problem is **data-limited**, and we verified it rather than assumed it:
- **Coupling is underdetermined.** ~32,640 pairwise terms vs only 8,000 train configs.
  A full all-pairs quadratic fits *train* to the noise floor but its *test* error
  explodes (textbook underdetermined overfit). Local degree-2 coupling (1,808 pairs) is
  the generalization sweet spot.
- **More CNN capacity / extra attention layers** did not give a stable gain within the
  1M budget; apparent improvements sat inside the run-to-run seed noise.
- **Counter-intuitively, Log is harder than Dipole** despite living in a *smaller*
  subspace: the bottleneck is the input→coefficient map, not the output rank, which is
  where the heavier (attention + 4-seed) Log model is spent.

## Result (verified)
Scoring the shipped `submission.csv` with the official NMSE formula against the
per-config frame-averaged truth of the 2,000 evaluation configs:

| condition | NMSE | (dB) |
|---|---|---|
| Dipole_pos1 | 0.00399 | −23.99 |
| Dipole_pos2 | 0.00482 | −23.17 |
| Dipole_pos3 | 0.00746 | −21.27 |
| Dipole_pos5 | 0.00350 | −24.56 |
| Log_pos1 | 0.00650 | −21.87 |
| Log_pos2 | 0.00723 | −21.41 |
| Log_pos3 | 0.01076 | −19.68 |
| Log_pos5 | 0.00517 | −22.86 |
| **mean (Final Score)** | **0.00618** | **−22.09** |

Equivalent public-leaderboard figure (un-normalized MSE, the number Kaggle displays):
**≈ 48,455**.

This is achieved purely by modelling, with **no evaluation-config channel used in
training or model selection** (the blend weights are chosen on a held-out train split).
It matches the level the dataset's own authors report: the BRISC paper (the team that
built this measurement campaign) fits the same model families — linear with bias (LMB),
neural network, random forest — on the same last-2,000 held-out configurations and
reaches roughly **−20 to −22 dB**, explicitly capped by a fixed uncontrollable component
("the frame of the RIS and other parts that are not controllable"), and notes that a
linear model with bias performs on par with a neural network. Our −22.09 dB sits at that
reported ceiling.

## Reproducibility & integrity
- **Reproducible.** `prep_data.py` builds the per-condition arrays from the official
  Kaggle CSVs; `DATA=data python the_silent_subcarriers_0.py reproduce` regenerates
  `submission.csv` **byte-for-byte** from the shipped weights (CPU-only).
- **Budget.** Each predictor ≤ 1M trainable params, counting **all** components — the
  CNN+quad **and** the macro-block refinement head (+1,632/condition: MB9 quad + MB4
  linear, reusing the quad's rank-32 SVD basis, so the basis is not double-counted) and
  the 32 convex-blend weights: **Log 976,916; Dipole 908,884** (both < 1M); ≈ 7.54M total
  across the 8 predictors < 8M. Verify with `python the_silent_subcarriers_0.py params`.
- **No leakage.** Trained only on the public-position training configs; the 2,000
  evaluation configs are never trained on (only their RIS bits are read as input), and
  the held-out positions {4, 6, 7, 8, 9} are never read at all.
