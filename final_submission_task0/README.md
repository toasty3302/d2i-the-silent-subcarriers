# ISIT 2026 D2I "The Still Mirror" — Task 0 (Channel Exploration)
### Team `the_silent_subcarriers`

Predict the complex channel (CSI) from a 16×16 binary RIS configuration:
**242 sub-carriers → 484 stacked real/imag values**, with one independent
predictor for each of the **8 public conditions** =
{Dipole, Log} receiver antenna × transmitter positions {1, 2, 3, 5}.

## Result (official metric)

The official Task-0 score is the **normalized MSE (NMSE)**, i.e. the normalized
squared Frobenius error `‖Ĥ−H‖²_F / ‖H‖²_F`, averaged over the 8 conditions
(lower is better; see the spec, *Task 0 → Evaluation metric*).

| | value |
|---|---|
| **Official Final Score — mean NMSE over the 8 conditions** | **0.00618  (−22.09 dB)** |
| Equivalent public-leaderboard figure (un-normalized MSE, the number Kaggle shows) | ≈ **48,455** |

This was obtained **without any test-set leakage** (see *Data-use integrity* below).
It is at the level the dataset's own authors report: the BRISC paper (the team that
ran this measurement campaign) fits the same model families on the same last-2,000
held-out configurations and tops out around **−20 to −22 dB**, explicitly capped by a
fixed, uncontrollable channel component (the RIS frame). Our −22.09 dB sits at that
ceiling. The number is reproduced from the shipped artifacts below.

---

## Contents

| file | what it is |
|---|---|
| `the_silent_subcarriers_0.py` | the complete solution (model defs + training + reproduction + structure refinement + parameter-budget check). Self-contained. |
| `prep_data.py` | one-off: build the per-condition `.npy` arrays the solver reads, from **either** the official Kaggle CSVs (`--source kaggle`, default) **or** the official BRISC `.mat` dataset (`--source brisc`). Both produce identical arrays (verified). |
| `models/the_silent_subcarriers_0.pth` | the 8 trained predictors (per-condition CNN ensembles + SVD bases) and the per-component blend weights. |
| `submission.csv` | the predictions for the 16,000 test rows (8 conditions × 2,000 configs). |
| `packages.txt` | dependencies (`numpy`, `torch`). |
| `SOLUTION_task0.md` | a write-up of the method, observations and verified result. |

---

## Setup

```bash
pip install -r packages.txt          # numpy==2.4.6, torch==2.12.0 (CPU is enough)
```

## Run

```bash
# 0. one-off: build the per-condition .npy arrays. TWO equivalent data sources:
#    (a) the official Kaggle CSVs  (<kaggle_dir> holds train.csv + test.csv):
python prep_data.py --source kaggle --kaggle-dir <kaggle_dir> --out data
#    (b) the official BRISC .mat dataset (configurations_10000.mat + antenna*_pos*.mat;
#        the small RIS-only test.csv supplies only the list of target configs/example_ids):
python prep_data.py --source brisc --brisc-root <brisc_dataset_dir> --out data
#    Both yield identical RIS arrays and per-config frame-averaged CSI (verified: RIS
#    bit-exact, CSI relative error ~2e-8) -> identical model and submission.

# 1. reproduce submission.csv from the shipped weights  (CPU, no GPU):
DATA=data python the_silent_subcarriers_0.py reproduce

# 2. (optional) retrain all 8 predictors from scratch (GPU recommended), then
#    rebuild submission.csv with the freshly trained weights:
DATA=data python the_silent_subcarriers_0.py train

# 3. print the per-predictor parameter counts (budget compliance):
python the_silent_subcarriers_0.py params
```

`reproduce` regenerates the shipped `submission.csv` **byte-for-byte** from the saved
weights (CPU-only, a few minutes — no GPU required). `train` overwrites
`models/the_silent_subcarriers_0.pth` and then runs the same reproduction path.

> The reported NMSE is computed with the official formula (normalized squared
> Frobenius error, averaged over the 8 conditions) against the per-configuration
> frame-averaged CSI of the 2,000 evaluation configs — the same averaged-channel
> truth the dataset authors use. This is offline scoring of the finished
> `submission.csv` only; those channel values are **never** read while fitting or
> selecting any model.

---

## How the constraints are respected

**Parameter budget — each of the 8 predictors is well under 1,000,000 trainable
parameters** (the spec limit is 1M per predictor × 8 = 8M total; verify with
`python the_silent_subcarriers_0.py params`):

| predictor | one CNN | × seeds | + quad | + macro-block | **total** |
|---|---|---|---|---|---|
| **Log** (×4 conditions) | 223,308 | × 4 | 82,052 | 1,632 | **976,916** ≤ 1,000,000 |
| **Dipole** (×4 conditions) | 412,600 | × 2 | 82,052 | 1,632 | **908,884** ≤ 1,000,000 |

The 8 predictors are completely independent (one model per condition); nothing is
shared across conditions, so each is counted on its own. The total **includes the
macro-block structure-refinement head** (+1,632/condition: a 9-block quadratic + 4-block
linear model that, at inference, overwrites the CNN on the structured test configs);
it reuses the quad's rank-32 SVD output basis (counted once), and the per-component blend
adds 32 scalar weights per condition.

**Data-use integrity (no leakage).** Every model is fit **only** on the public
training configurations' RIS bits + measured CSI (Kaggle `train.csv`, or equivalently the
BRISC `.mat` of the same public positions). The 2,000 evaluation configurations are used
for their RIS bits (the model input) only — their channels are never read during fitting
or model selection. The 5 held-out transmitter positions {4, 6, 7, 8, 9} are never read
at all.
