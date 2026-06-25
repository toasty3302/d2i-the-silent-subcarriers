# Task 2 - Phase Optimization
### Team `the_silent_subcarriers`

Task 2 asks us to propose a binary 256-element RIS configuration for each evaluation condition. The goal is to maximize Gaussian mutual information on the center subcarrier.

The scoring convention used here is the normalized one from the baseline:

```text
I(s) = log2(1 + rho * |h_norm(s)|^2), rho = 10, SNR = 10 dB
h_norm = h / sqrt(mean(|h|^2))
```

All reported MI values in this package use those normalized bits. The much larger raw `log2(1 + |h|^2)` numbers are not the leaderboard units.

## What to submit

The spec leaves one important detail open: the final score may use held-out ground-truth measurements or an official evaluation oracle. The k-nearest-neighbor Hamming approximation is described as a diagnostic. Because those regimes treat off-measured configurations differently, this package contains two config sets.

| file | idea | mean MI | recommendation |
| --- | --- | ---: | --- |
| `proposed_configs.json` | best measured config per condition | 5.94 true normalized bits | Primary submission |
| `proposed_configs_aggressive.json` | affine phase-sweep optimum | 7.95 surrogate bits, about 7.88 by independent oracle | Use only if arbitrary configs are scored by a smooth oracle |

The primary file is the safer default. Each proposed config is one of the measured configurations, so its MI is known and it remains valid under lookup, kNN-style diagnostics, or an oracle.

The aggressive file is a real optimization result, but it is far from the measured manifold: 38 to 101 Hamming bits from every measured config. It is attractive only if the official scorer can evaluate arbitrary binary configs smoothly.

## Files

| file | purpose |
| --- | --- |
| `the_silent_subcarriers_2.py` | Condition-agnostic model and optimization procedure. |
| `proposed_configs.json` | Primary robust config set. |
| `proposed_configs_aggressive.json` | Alternative aggressive config set. |
| `models/the_silent_subcarriers_2.pkl` | Saved trained-model artifact (spec req #2): the 8 per-condition affine CSI surrogates (`h_hat = beta0 + beta[1:]·s`), frozen after fitting. 257 complex coeffs/condition = 4,112 real DoF total ≪ 20,000,000. The aggressive configs are reproducible from this artifact via `phase_sweep_optimum`. |
| `SOLUTION_task2.md` | Method notes, scorer ambiguity, and validation checks. |
| `packages.txt` | Dependencies. |

## Run

```bash
pip install -r packages.txt

# Primary robust submission
python the_silent_subcarriers_2.py

# Alternative aggressive submission
python the_silent_subcarriers_2.py --mode aggressive

# Explicit dataset path
python the_silent_subcarriers_2.py --data /path/to/ISIT2026-challenge-dataset --out proposed_configs.json

# Also save the fitted affine surrogates to models/ (spec req #2)
python the_silent_subcarriers_2.py --mode aggressive --save-models
```

The script first tries the official `task2_loader`. If that is not available, it discovers the `antenna*_pos*.mat` files in the dataset directory. It does not hard-code the public positions, so the same procedure can run on private grading conditions supplied by the loader.

## Public-condition results

| condition | primary best-measured | aggressive affine optimum | min Hamming distance from aggressive to measured |
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

The aggressive numbers are surrogate predictions. An independently fit quadratic oracle scores the same aggressive configs at about `7.88` mean normalized bits, while the kNN diagnostic is about `5.1`.

## Constraint handling

- All emitted RIS states are binary vectors in `{0,1}^256`.
- The primary best-measured selection has 0 trainable parameters.
- The affine surrogate has 257 complex parameters per condition, far below the 20M limit.
- Development used only public positions `{1, 2, 3, 5}`.
- The procedure is condition-agnostic and refits per condition from whatever measured table the official loader provides.
