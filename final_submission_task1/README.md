# ISIT 2026 D2I "The Still Mirror" — Task 1 (Mutual Information Estimation)
### Team `the_silent_subcarriers`

For each example — one (antenna type, transmitter position, RIS configuration) on
the center OFDM subcarrier — estimate the **mutual information (in bits)** between
the channel input and the channel output, under a **Gaussian-input** assumption,
from **256 noisy complex CSI samples** (AWGN added to reach SNR = 10 dB).
Official metric: **RMSE in bits** between predicted and held-out true MI (lower is better).

**Headline result (leakage-free, grouped-by-configuration 5-fold CV on the 120 random
training examples): RMSE ≈ 0.047 bits** for the closed-form estimator that this
solution is built around. For reference, the competition's MINE/SMILE-style neural
baseline (`Task_1_MI_estimation.py`, sample estimator) reaches ≈ **0.13 bits** mean
held-out RMSE on the same data — our estimator is roughly **2.5× lower error**.
See `SOLUTION_task1.md` for the full method, the validation table, and an honest
note on the shipped `residual_scale` hyperparameter.

---

## Contents

| file | what it is |
|---|---|
| `the_silent_subcarriers_1.py` | the complete, self-contained solution: data loading, the closed-form MI estimator + ridge residual correction, leakage-free cross-validation, the parameter-budget check, and submission writing. Fits at run time — no separate weight file. |
| `submission.csv` | the shipped predictions for the 72 test examples (`example_id,mi_bits`). |
| `data/` | the official Task 1 CSV package, bundled so the package is self-contained: `train.csv` (168 rows, labeled), `test.csv` (72 rows, unlabeled), `sample_submission.csv` (required output format), `metaData.csv` (task description). |
| `SOLUTION_task1.md` | method write-up: estimator, why it beats the MINE/SMILE baseline, the metric, leakage-free validation, parameter compliance. |
| `packages.txt` | dependencies (`numpy` only). |

---

## Setup

```bash
pip install -r packages.txt          # numpy only; CPU is enough
```

## Run

```bash
# Regenerate submission.csv from the bundled data (CPU, < 1 second).
# Defaults read ./data and write ./submission.csv; runnable from any directory.
python the_silent_subcarriers_1.py

# Print the leakage-free CV report only, without rewriting the file:
python the_silent_subcarriers_1.py --output /tmp/throwaway.csv

# Print the per-model parameter count (budget compliance):
python the_silent_subcarriers_1.py --params
```

Running with no arguments prints the cross-validation table and rewrites
`submission.csv`. The regenerated file **matches the shipped `submission.csv`
bit-for-bit** (max absolute difference 0.0). To point at a different copy of the
official CSVs, pass `--data-dir <dir>`.

---

## How the constraints are respected

* **Parameter budget.** The spec (Task 1, Submission) allows **8 models of at most
  2,000,000 parameters each**. Our per-condition model is a closed-form MI estimator
  plus a ridge residual with **11 coefficients (31 learned scalars including the
  feature-standardization statistics)** — about five orders of magnitude under the
  limit. Verify with `python the_silent_subcarriers_1.py --params`.
* **Submission format.** Output columns are exactly `example_id,mi_bits`, 72 rows,
  in the same order as `data/sample_submission.csv`. Every `example_id` is present
  exactly once; all predicted values are finite and non-negative (range ≈ 5.7–8.1 bits).
* **Data-use integrity / no leakage.** The solution reads only the released Task 1
  CSV package. The residual model is fit only on `train.csv` labels; `test.csv`
  contains no `mi_bits` column and no held-out MI is ever used for fitting. The
  reported metric comes from cross-validation **grouped by RIS configuration id**,
  so no configuration is simultaneously trained on and validated on.
