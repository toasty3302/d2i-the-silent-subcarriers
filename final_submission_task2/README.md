# Task 2 — Phase Optimization (team `the_silent_subcarriers`)

Propose a **binary 256-element RIS configuration** `s ∈ {0,1}^256` per evaluation condition
that maximizes the center-subcarrier Gaussian mutual information

```
I(s) = log2(1 + ρ · |h_norm(s)|²),    ρ = 10^(SNR_dB/10) = 10   (SNR = 10 dB)
```

where the channel is **power-normalized** as `h_norm = h / power_scale`,
`power_scale = sqrt(mean(|h|²))` over that condition's measured configs. This is the official
convention (`Task_2_Phase-Opt.py::gaussian_mi_bits`); **all numbers in this package are in these
normalized bits** (the alternative "raw" `log2(1+|h|²)` units, ~28 bits, are a units mismatch for
the leaderboard and are *not* used here).

## What to submit

The scoring rule the organizers use is not published. Per the spec, the official evaluation
scores the submitted config "using the held-out ground-truth measurements **or** the official
evaluation oracle"; the 10-nearest-neighbor (Hamming) approximation is described only as a
baseline **diagnostic**. We therefore ship two submissions and recommend the robust one.

| file | configs | mean MI (normalized bits) | recommendation |
|---|---|---|---|
| **`proposed_configs.json`** | best-measured per condition | **5.94 (true MI)** | **PRIMARY — submit this.** Valid under every scoring rule. |
| `proposed_configs_aggressive.json` | exact affine phase-sweep optimum | 7.95 (surrogate) / ≈7.88 (independent oracle) | Alternative — only if organizers confirm a smooth evaluation oracle. |

The primary configs are **measured** configurations, so their MI is a *known, exact* value and is
scorable under any rule (exact under ground-truth lookup, on-manifold under an oracle, ~itself
under the kNN diagnostic). The aggressive configs are model optima that sit **38–101 Hamming bits**
from every measured config: they reach ~7.9 bits **only** if the official oracle is smooth and
agrees with our surrogate; under ground-truth lookup they are not a measured config, and under the
kNN diagnostic they regress below the best-measured floor. The aggressive set is a regime-contingent
bet, not a free gain — hence robust-by-default.

## How to run

```bash
pip install -r packages.txt

# PRIMARY submission (best-measured, recommended):
python the_silent_subcarriers_2.py                       # -> proposed_configs.json

# ALTERNATIVE submission (affine phase-sweep optimum; smooth-oracle regime only):
python the_silent_subcarriers_2.py --mode aggressive     # -> proposed_configs.json  (aggressive configs)

# point at the dataset explicitly if needed:
python the_silent_subcarriers_2.py --data /path/to/ISIT2026-challenge-dataset --out proposed_configs.json
```

`the_silent_subcarriers_2.py` first tries `import task2_loader` (the official Task 2 loader: `evaluation_conditions()`
+ `load_condition()`), so it runs **unchanged on the private evaluation conditions** at grading
time. If that import fails it falls back to **discovering every `antenna*_pos*.mat`** in the dataset
directory (it does not hard-code the public positions, so private conditions are handled the same
way). For each condition it computes the averaged center-subcarrier CSI, fits a small per-condition
surrogate, and emits both the best-measured config and the affine optimum; `--mode` selects which one
is written as the submission.

The shipped `proposed_configs*.json` were produced by this exact procedure; re-running
`python the_silent_subcarriers_2.py` regenerates `proposed_configs.json` byte-for-byte (verified by streaming the raw
`.mat` measurements end-to-end).

## Results (normalized convention, ρ = 10)

| condition | best-measured (PRIMARY) | affine optimum (ALT, surrogate) | min Hamming of ALT to any measured config |
|---|---|---|---|
| Dipole_1 | 5.97 | 8.52 | 99 |
| Dipole_2 | 5.08 | 7.23 | 97 |
| Dipole_3 | 5.39 | 7.77 | 98 |
| Dipole_5 | 7.44 | 8.12 | 54 |
| Log_1 | 5.82 | 8.61 | 94 |
| Log_2 | 5.72 | 8.28 | 92 |
| Log_3 | 5.69 | 8.28 | 101 |
| Log_5 | 6.40 | 6.79 | 38 |
| **mean** | **5.94** | **7.95** | — |

(The aggressive column is the affine surrogate's own prediction. A held-out, independently fit
quadratic oracle scores the same aggressive configs at **mean ≈ 7.88** normalized bits and the kNN
diagnostic at ≈ 5.1 — see `SOLUTION_task2.md`.) The 8 conditions here are the public set
{Dipole, Log} × positions {1, 2, 3, 5}; at grading the same code runs on whatever the loader returns.

## Compliance

- **1-bit binary only.** Every emitted config is in `{0,1}^256` (verified). The RIS is physically
  1-bit; no continuous / multi-bit "phase beamforming" is proposed.
- **≤ 20M parameters.** The per-condition model is an affine CSI surrogate: **257 complex
  parameters** (1 bias + 256 linear). The best-measured selection has **0** trainable parameters.
  Far below the 20,000,000-parameter Task 2 limit.
- **Condition-agnostic.** Nothing is hard-coded to the 8 public conditions. The submission is a
  model + optimization procedure re-run by the official loader; it fits per condition on whatever
  measured table the loader provides (public now, **private at grading**).
- **No private-position use.** Only the public transmitter positions {1, 2, 3, 5} were used in
  development. The held-out positions {4, 6, 7, 8, 9} were never loaded, inferred, or scored against.
- **No test-config answer leakage.** Each condition's surrogate is fit only on that condition's
  legitimately public measured configs. The primary submission is the condition's own best
  *measured* config (a known measurement, not an injected held-out answer); the aggressive
  submission is the surrogate's argmax. Neither emits memorized held-out ground truth as a prediction.

## Files

- `the_silent_subcarriers_2.py` — the condition-agnostic model + optimization procedure (the submission artifact).
- `proposed_configs.json` — **PRIMARY**: best-measured config + normalized MI per condition.
- `proposed_configs_aggressive.json` — ALTERNATIVE: affine phase-sweep optimum per condition.
- `SOLUTION_task2.md` — method write-up (surrogate, exact phase-sweep optimum, why richer models do
  not help, and the scoring-ambiguity rationale).
- `packages.txt` — Python dependencies.
