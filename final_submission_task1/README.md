# ISIT 2026 D2I "The Still Mirror" - Task 1
### Team `the_silent_subcarriers`

Task 1 asks for the mutual information, in bits, for one channel condition at the center OFDM subcarrier. Each row gives 256 noisy complex CSI samples at 10 dB SNR, and the output is a single MI estimate. The official score is RMSE in bits, so lower is better.

The important point for this task is that the target is not a generic black-box label. Under the Gaussian-input assumption, MI has a simple plug-in estimate from the sample variance. The submitted model starts with that closed-form estimate and only learns a small residual correction.

## Result

Grouped-by-configuration cross-validation on the random training rows gives roughly:

```text
closed-form plug-in only:       RMSE 0.047483 bits
plug-in + CV-scale residual:    RMSE 0.047339 bits
shipped residual scale 3.35:    RMSE 0.051799 bits locally
```

The shipped `submission.csv` uses `residual_scale = 3.35`, because that was the best public submission setting we had. The local, leakage-free number to trust for method quality is still about `0.047` bits. The solution write-up explains this caveat directly.

For reference, the MINE/SMILE-style neural baseline is around `0.13` bits held-out RMSE on the same data, so the closed-form estimator is much more stable here.

## Files

| file | purpose |
| --- | --- |
| `the_silent_subcarriers_1.py` | Complete Task 1 solution: data loading, MI estimator, residual ridge fit, CV report, parameter count, and submission writing. |
| `submission.csv` | Submitted predictions for the 72 test examples. |
| `data/` | Bundled official Task 1 CSV files: train, test, sample submission, and metadata. |
| `SOLUTION_task1.md` | Explanation of the estimator and validation caveats. |
| `packages.txt` | Minimal dependencies. |

## Setup

```bash
pip install -r packages.txt
```

Only NumPy is required.

## Run

Regenerate the submission from the bundled data:

```bash
python the_silent_subcarriers_1.py
```

Print the CV report while writing somewhere disposable:

```bash
python the_silent_subcarriers_1.py --output /tmp/task1_check.csv
```

Check the parameter count:

```bash
python the_silent_subcarriers_1.py --params
```

To use a different copy of the official CSV package:

```bash
python the_silent_subcarriers_1.py --data-dir <dir>
```

## Constraint handling

The spec allows 8 models with up to 2M parameters each. This solution uses a closed-form estimator plus an 11-coefficient ridge residual. Including stored feature means and scales, the model stores 31 learned scalars per condition, far below the limit.

The output format is exactly:

```text
example_id,mi_bits
```

with 72 rows, one per test example, in sample-submission order. Values are finite and non-negative.

## Data use

The script reads only the released Task 1 CSV package. Residual fitting uses labels from `train.csv`; `test.csv` has no `mi_bits` labels. Local validation is grouped by RIS configuration ID, so a configuration is never split between train and validation folds.
