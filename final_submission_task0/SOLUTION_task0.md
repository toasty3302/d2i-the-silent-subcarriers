# Task 0 solution summary
### Team `the_silent_subcarriers`

## What the task asks for

For each public condition, the input is a binary 16 x 16 RIS state and the target is the complex CSI over 242 subcarriers. The output is scored with normalized MSE, averaged over 8 conditions:

```text
{Dipole, Log} x positions {1, 2, 3, 5}
```

Lower NMSE is better. Each condition must stay under 1M trainable parameters.

## Main approach

The final predictor is a small blend of two views of the same channel:

```text
prediction = per-component convex blend of (quadratic ridge, CNN ensemble)
then structured macro-block configs are refined by a tiny closed-form model
```

The quadratic model is there because the channel is close to affine in the RIS reflection states, with a useful amount of local coupling. The CNN ensemble is there because the quadratic model misses some nonlinear residual structure. They make different errors, so blending them by SVD component works better than choosing one model globally.

## Details that mattered

**Local coupling.** A full all-pairs quadratic would have 32,640 pair terms, which is too many for 8,000 training configurations. Instead we keep the 1,808 nearest-neighbor pair products on the panel. That captures local RIS coupling without giving the model enough freedom to memorize.

**Low-rank output.** The 484-dimensional real/imag CSI output lives in a much smaller subspace. All predictors decode through a fixed rank-32 SVD basis from the training channels. This regularizes the output and keeps the parameter count under control.

**Different model sizes for Dipole and Log.** Log needed more model diversity, so it uses 4 CNN seeds and one attention layer. Dipole was easier, so 2 seeds without attention were enough.

**Structured config refinement.** The first RIS configurations are not random: the 4-block and 9-block macro layouts are much lower-dimensional. For those structured configs, a tiny model fit on the matching training family is more reliable than asking the CNN to extrapolate. The refinement affects 104 configs per condition and is counted in the parameter budget.

## Result

Scoring the shipped `submission.csv` with the official NMSE formula gives:

| condition | NMSE | dB |
| --- | ---: | ---: |
| Dipole_pos1 | 0.00399 | -23.99 |
| Dipole_pos2 | 0.00482 | -23.17 |
| Dipole_pos3 | 0.00746 | -21.27 |
| Dipole_pos5 | 0.00350 | -24.56 |
| Log_pos1 | 0.00650 | -21.87 |
| Log_pos2 | 0.00723 | -21.41 |
| Log_pos3 | 0.01076 | -19.68 |
| Log_pos5 | 0.00517 | -22.86 |
| mean | 0.00618 | -22.09 |

The corresponding public-leaderboard-style unnormalized MSE is about `48,455`.

## What we tried and did not keep

A few things looked tempting but did not hold up:

- Full all-pairs quadratic features fit training very well but generalize badly.
- Larger CNNs and extra attention layers did not give a stable gain inside the 1M limit.
- More output rank increased capacity without improving the final score enough.
- Treating all conditions with the same architecture wasted parameters on Dipole and under-spent on Log.

The final version is deliberately not the largest model we could fit. It is the mix that generalized best while staying under the per-condition cap.

## Reproducibility and integrity

`prep_data.py` creates the arrays from the official data, and

```bash
DATA=data python the_silent_subcarriers_0.py reproduce
```

regenerates the shipped `submission.csv` from the saved weights. Parameter counts can be checked with:

```bash
python the_silent_subcarriers_0.py params
```

All training and blending uses public training configurations only. Evaluation channels and held-out positions are not used for fitting or model selection.
