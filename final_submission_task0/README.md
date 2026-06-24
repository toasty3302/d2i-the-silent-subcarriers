# ISIT 2026 D2I "The Still Mirror" - Task 0
### Team `the_silent_subcarriers`

Task 0 is the channel-prediction task. For each public receiver/position condition, the model gets a 16 x 16 binary RIS configuration and predicts the complex CSI over 242 subcarriers. In the submitted files, the complex output is represented as 484 real values: all real parts and all imaginary parts.

There are 8 public conditions:

```text
{Dipole, Log} x positions {1, 2, 3, 5}
```

We train one independent predictor for each condition. The official metric is normalized MSE (NMSE), averaged over the 8 conditions, so lower is better.

## Result

| metric | value |
| --- | ---: |
| Official-style mean NMSE over the 8 public conditions | 0.00618 |
| Same value in dB | -22.09 dB |
| Equivalent public-leaderboard unnormalized MSE | about 48,455 |

This result is from the shipped `submission.csv`. It was produced without using held-out position data or test channels during model fitting. The score is close to the ceiling reported by the BRISC dataset authors for this measured setup, where a fixed uncontrolled channel component limits the benefit of larger models.

## Files

| file | purpose |
| --- | --- |
| `the_silent_subcarriers_0.py` | Main Task 0 runner. It contains the model definitions, training path, reproduction path, structure refinement, and parameter counting. |
| `prep_data.py` | Converts the official Kaggle CSVs or the BRISC `.mat` files into the per-condition arrays used by the runner. |
| `models/the_silent_subcarriers_0.pth` | Saved weights for the 8 predictors. |
| `submission.csv` | Submitted predictions for the 16,000 test rows. |
| `packages.txt` | Python dependencies. |
| `SOLUTION_task0.md` | Method notes, results, and integrity checks. |

## Setup

```bash
pip install -r packages.txt
```

## Reproduce the submission

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

## Constraint handling

Each condition has its own predictor and stays below the 1M-parameter limit:

| predictor family | one CNN | seeds | quad | macro-block | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Log conditions | 223,308 | 4 | 82,052 | 1,632 | 976,916 |
| Dipole conditions | 412,600 | 2 | 82,052 | 1,632 | 908,884 |

The parameter count includes the CNN ensemble, the quadratic model, the macro-block refinement head, and the per-component blend weights.

## Data use

The models are fit only on public-position training configurations. The evaluation rows are used for their RIS bits only, which are the model input. Their channel values are not used for training, blending, model selection, or debugging. Held-out transmitter positions `{4, 6, 7, 8, 9}` are not read.
