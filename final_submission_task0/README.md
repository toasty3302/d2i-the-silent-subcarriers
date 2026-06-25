# ISIT 2026 D2I "The Still Mirror" - Task 0
### Team `the_silent_subcarriers`

Task 0 is the channel-prediction task. For each public receiver/position condition, the model gets a 16 x 16 binary RIS configuration and predicts the complex CSI over 242 subcarriers. In the submitted files, the complex output is represented as 484 real values: all real parts and all imaginary parts.

There are 8 public conditions:

```text
{Dipole, Log} x positions {1, 2, 3, 5}
```

We train one independent predictor for each condition. The official metric is normalized MSE (NMSE), averaged over the 8 conditions.

## Constraint handling

Each condition has its own predictor and stays below the 1M-parameter limit:

| predictor family | one CNN | seeds | quad | macro-block | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Log conditions | 223,308 | 4 | 82,052 | 1,632 | 976,916 |
| Dipole conditions | 412,600 | 2 | 82,052 | 1,632 | 908,884 |

The parameter count includes the CNN ensemble, the quadratic model, the macro-block refinement head, and the per-component blend weights.

## Result

| metric | value |
| --- | ---: |
| mean NMSE over the 8 public conditions | 0.00618 |
| Same value in dB | -22.09 dB |
| mean raw MSE over the 8 public conditions | about 48,455 |

This result is from the shipped `submission.csv`.

## Setup

```bash
pip install -r packages.txt
```

## Run

First build the per-condition arrays. Either source is acceptable; they were checked to produce the same arrays.

```bash
# From the official Kaggle files: train.csv and test.csv
python prep_data.py --source kaggle --kaggle-dir <kaggle_dir> --out data

# Or from the BRISC .mat files
python prep_data.py --source brisc --brisc-root <brisc_dataset_dir> --out data
```

Then regenerate `submission.csv` from the shipped model weights:

```bash
DATA=data python the_silent_subcarriers_0.py reproduce
```

Optional commands:

```bash
# Retrain all 8 predictors, then rebuild submission.csv
DATA=data python the_silent_subcarriers_0.py train

# Print the per-condition parameter counts
python the_silent_subcarriers_0.py params
```

The `reproduce` command is CPU-only and regenerates the shipped submission from the saved weights. Training from scratch is GPU-recommended because the CNN ensembles are the slow part.

## Main approach

The final predictor is a small blend of two views of the same channel:

```text
prediction = per-component convex blend of (quadratic ridge, CNN ensemble)
then structured macro-block configs are refined by a tiny closed-form model
```

The quadratic model is there because to model the affine relation in the RIS reflection states. The CNN ensemble is there to model the nonlinear residual structure that the quadratic model misses. They make different errors, so blending them by SVD component works better than choosing one model globally.

Additionally, if the input has a macroblock configuration, we use a smaller closed-ridge regression model. The combined number of parameters of all the modes is less than the 1M limit.   

## Observations

**Local coupling.** A full all-pairs quadratic would have 32,640 pair terms, which is too many for 8,000 training configurations. Instead we keep the 1,808 nearest-neighbor pair products on the panel. That captures local RIS coupling without giving the model enough freedom to memorize.

**Low-rank output.** The 484-dimensional real/imag CSI output lives in a much smaller subspace. All predictors decode through a fixed rank-32 SVD basis from the training channels. This regularizes the output and keeps the parameter count under control.

**Different model sizes for Dipole and Log.** Log needed more model diversity, so it uses 4 CNN seeds and one attention layer. Dipole was easier, so 2 seeds without attention were enough.

**Structured config refinement.** The first RIS configurations are not random: the 4-block and 9-block macro layouts are much lower-dimensional. For those structured configs, a tiny model fit on the matching training family is more reliable than asking the CNN to extrapolate. The refinement affects 104 configs per condition and is counted in the parameter budget.

## What we tried and did not keep

A few things looked tempting but did not hold up:

- Full all-pairs quadratic features fit training very well but generalize badly.
- Larger CNNs and extra attention layers did not give a stable gain inside the 1M limit.
- More output rank increased capacity without improving the final score enough.
- Treating all conditions with the same architecture wasted parameters on Dipole and under-spent on Log.

The final version is deliberately not the largest model we could fit. It is the mix that generalized best while staying under the per-condition cap.
