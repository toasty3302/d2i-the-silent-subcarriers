# Task 2 - Phase Optimization
### Team `the_silent_subcarriers`

Task 2 asks us to propose a binary 256-element RIS configuration for each evaluation condition. The goal is to maximize Gaussian mutual information on the center subcarrier.

The scoring convention used here is the normalized one from the baseline:

```text
I(s) = log2(1 + rho * |h_norm(s)|^2), rho = 10, SNR = 10 dB
h_norm = h / sqrt(mean(|h|^2))
```

## The two modes (robust vs aggressive)

The script emits a submission in one of two modes. They differ in **where the proposed
config lives relative to the measured data**, which is the whole question for this task.

| | **robust** (default) | **aggressive** (`--mode aggressive`) |
|---|---|---|
| What it submits | the **best *measured* config** for the condition | the **exact affine-surrogate optimum** (`argmax_s \|b + wᵀs\|²`) |
| Trainable params | 0 (pure lookup) | 257 complex (one affine surrogate/condition) |
| Config location | **on** the measured manifold | **off** it — 38–101 Hamming bits from any measured config |
| MI value | a **known, true measured** MI (≈5.94 mean) | a **predicted** MI (≈7.95 mean), only realized if it scores |
| Valid when the scorer is… | a lookup, the kNN diagnostic, **or** an oracle (always) | **only** a smooth oracle over arbitrary binary configs |

**Why aggressive is different — and why it isn't the default.** The affine optimum predicts
~2 more bits than the best measured config, but it is a config the hardware was never measured
at (tens of Hamming bits off-manifold). That gain is only real if the official scorer is a
*smooth oracle* that evaluates arbitrary configurations. Under the other two scoring
possibilities the spec leaves open, aggressive is actively worse: a **measured-lookup** scorer
may not be able to score an unmeasured config at all, and the **kNN diagnostic** drags it back
*below* the best-measured floor (~5.1 bits). Robust takes the known-true measured config, so it
is valid — and never below the floor — under *every* scorer. We therefore ship robust as the
primary submission and include `proposed_configs_aggressive.json` as the alternative to use only
if the organizers confirm arbitrary-config oracle scoring. (Detailed reasoning and per-condition
numbers are in the sections below.)

## Constraint handling

- All emitted RIS states are binary vectors in `{0,1}^256`.
- The primary best-measured selection has 0 trainable parameters.
- The affine surrogate has 257 complex parameters per condition, far below the 20M limit.
- Development used only public positions `{1, 2, 3, 5}`.
- The procedure is condition-agnostic and refits per condition from whatever measured table the official loader provides.

## Setup 

```bash
pip install -r packages.txt
```

## Run

# Primary robust submission
python the_silent_subcarriers_2.py

# Alternative aggressive submission
python the_silent_subcarriers_2.py --mode aggressive

# Explicit dataset path
python the_silent_subcarriers_2.py --data /path/to/ISIT2026-challenge-dataset --out proposed_configs.json

# Also save the fitted affine surrogates to models/ (spec req #2)
python the_silent_subcarriers_2.py --mode aggressive --save-models

## The core issue

The spec says the final evaluation may use held-out ground-truth measurements or an official evaluation oracle. The 10-nearest-neighbor Hamming approximation is described as a diagnostic. Those possibilities lead to different decisions:

- If scoring is a measured-config lookup, the safest choice is a measured config.
- If scoring is a kNN diagnostic, staying near measured configs is also safer.
- If scoring is a smooth oracle for arbitrary binary configs, a surrogate optimum can win.

That is why the package includes both a primary robust submission and an aggressive alternative.

## Primary submission: best measured config

For each condition, the primary `proposed_configs.json` chooses the measured RIS configuration with the highest observed normalized MI. This has a known value and does not depend on extrapolation.

Mean public-condition MI:

```text
best measured configs: 5.94 normalized bits
```

This is the recommendation because it is valid under every plausible scorer.

## Alternative submission: exact affine optimum

A 1-bit RIS suggests a simple affine channel model:

```text
h(s) ~= b + w^T s
```

For this model, maximizing `|b + w^T s|^2` over binary `s` can be solved exactly. For a target phase, include all elements whose rotated contribution has positive real part. As the phase sweeps from 0 to pi, the selected subset changes only at finitely many breakpoints. Checking the midpoints of those intervals gives the global optimum of the affine surrogate.

That aggressive optimum reaches:

```text
affine surrogate mean:           7.95 normalized bits
independent quadratic oracle:    about 7.88 normalized bits
kNN diagnostic:                  about 5.1 normalized bits
```

So the aggressive config is credible if the scorer is a smooth oracle, but it is risky because it is 38 to 101 Hamming bits away from any measured configuration.

## Why not just use the aggressive one?

Because two surrogate models agreeing is still not the same as the organizer's hidden scorer agreeing. The aggressive configs are far off the measured manifold. If the official scorer expects a measured config, they may be unscorable; if it uses the kNN diagnostic, they fall below the best-measured floor.

So the default is robust rather than speculative. The aggressive file is included in case the organizers explicitly confirm arbitrary-config oracle scoring.

## Public-condition numbers

| condition | best measured | affine optimum | Hamming distance from affine optimum to nearest measured config |
| --- | ---: | ---: | ---: |
| Dipole_1 | 5.97 | 8.52 | 99 |
| Dipole_2 | 5.08 | 7.23 | 97 |
| Dipole_3 | 5.39 | 7.77 | 98 |
| Dipole_5 | 7.44 | 8.12 | 54 |
| Log_1 | 5.82 | 8.61 | 94 |
| Log_2 | 5.72 | 8.28 | 92 |
| Log_3 | 5.69 | 8.28 | 101 |
| Log_5 | 6.40 | 6.79 | 38 |
| mean | 5.94 | 7.95 | |

## What did not improve the robust choice

- A distance-2 quadratic surrogate fits held-out measured configs better, but its off-manifold optimum is less trustworthy.
- Neural and graph-style surrogates did not beat the simple quadratic on held-out NMSE in a way that improved the proposed config.
- All-pairs interaction terms are too underdetermined for about 8,000 configs and overfit badly.
- Large search procedures still tended to either return the affine optimum or move away from the measured manifold without improving robust score.
- Denoising does not change much because the per-config CSI is already an average over many frames.

## Compliance

- Every emitted config is binary.
- Primary mode uses 0 trainable parameters.
- Aggressive mode uses a 257-complex-parameter affine surrogate per condition.
- Both are far below the 20M limit.
- The code is condition-agnostic and fits per condition from the measured table supplied by the loader.
- No held-out private positions were used in development.

Run:

```bash
python the_silent_subcarriers_2.py
python the_silent_subcarriers_2.py --mode aggressive
```
