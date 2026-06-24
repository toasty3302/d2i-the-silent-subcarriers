# ISIT 2026 D2I "The Still Mirror" - Task 3
### Team `the_silent_subcarriers`

This is our Task 3 submission for secret-key generation. The task is to be Eve and predict the key bits Alice gets from her own noisy channel, while Eve only sees her own channel from a different transmitter position under the same RIS configuration. The metric is mean bit mismatch rate (BMR), so lower is better, and the submitted model has to stay under 40 million parameters.

The headline local mock-test result is:

```text
mean BMR = 0.1965 over the 12 public Alice/Eve pairs
noise floor = 0.099
single decode mean = 0.2223
isotropic majority vote mean = 0.2002
bootstrap majority vote mean = 0.1965
```

For comparison, an official-style Eve-to-Alice CSI ridge baseline that ignores the RIS configuration scores about `0.38` BMR on the same 12 conditions. A configuration-only version of our model is already near `0.27`. The final method improves further by using Eve's residual CSI information and by decoding with a train-calibrated Bayesian majority vote instead of trusting one point estimate.

All numbers above can be reproduced with `the_silent_subcarriers_3.py`. Add `--baseline-ridge` to print the reference ridge baseline as well.

## Files

| file | purpose |
| --- | --- |
| `the_silent_subcarriers_3.py` | Complete Task 3 solution. It includes the extractor, model fitting, calibration, majority-vote decoding, parameter counting, and CLI. |
| `models/the_silent_subcarriers_3.pkl` | Saved public-condition models produced by `--save-models`; about 1.03M predictor parameters per condition. |
| `config.yaml` | Dataset path and public-position settings used by the script. |
| `packages.txt` | Minimal dependencies. Torch is optional and only speeds up Viterbi decoding when CUDA is available. |
| `SOLUTION_task3.md` | More detailed explanation of the approach, checks, and experiments. |

## Setup

```bash
pip install -r packages.txt
```

The script expects the BRISC dataset at the path in `config.yaml`, by default:

```text
../ISIT2026-challenge-dataset
```

That folder should contain the `antenna{Dipole,Log}_pos*.mat` files and `configurations_10000.mat`.

## Reproduce the main result

Run the full public mock test over both antennas, Alice positions 1 and 2, and Eve positions 1, 2, 3, and 5. Same-position Alice/Eve pairs are skipped.

```bash
python the_silent_subcarriers_3.py --antenna-types Dipole Log \
       --alice-positions 1 2 --eve-positions 1 2 3 5
```

Expected output includes `final_average_bmr = 0.1965...` for the 12 public conditions listed in `SOLUTION_task3.md`.

To also compute the reference ridge baseline:

```bash
python the_silent_subcarriers_3.py --antenna-types Dipole Log \
       --alice-positions 1 2 --eve-positions 1 2 3 5 \
       --baseline-ridge
```

The full run is CPU-bound because the majority-vote decoder runs many Viterbi passes. On a single CPU core it can take roughly 20 to 60 minutes. Fitting itself is closed-form and quick; the decoding loop is the expensive part.

## Other useful commands

Save or refresh the model pickle:

```bash
python the_silent_subcarriers_3.py --save-models
```

Run one condition:

```bash
python the_silent_subcarriers_3.py --antenna-type Dipole --alice-positions 1 --eve-positions 2
```

Run a faster smoke test by using fewer configurations:

```bash
python the_silent_subcarriers_3.py --antenna-type Dipole --alice-positions 1 --eve-positions 2 --n-configs 600
```

The smoke-test BMR is usually higher than the full-data result, but it is useful for checking that the extractor, model fit, and majority-vote path are working.

## Constraint handling

Each condition uses one folded linear predictor:

```text
h_A_hat = design_quad(RIS) @ W_config + eve_csi @ W_eve
```

The folded predictor has `1,025,596` parameters per condition. For the 12 public conditions, that is about `12.3M` predictor parameters. The bootstrap majority vote also stores a held-out error pool as decode-time uncertainty calibration rather than as predictor weights. Even if that pool is counted conservatively, the total is `16.95M`, well under the `40M` limit.

All fitting and calibration use only training configurations from public positions. Evaluation Alice CSI is not read during fitting. Private positions are not used as auxiliary training data; they are only loaded if an evaluator explicitly asks for that condition. The majority-vote scale is selected on a train-only hold-out slice, using Alice's public 35 dB noise model.

## Extractor check

Alice's extractor is 8-level Gray quantisation followed by a rate-1/3, K=5 Viterbi decoder. One easy mistake is the Sionna LLR sign convention: the reference call `llr = (1 - 2*bit)*10` means the decoder is effectively applied to the complement of the raw Gray bits. The NumPy extractor in this package was checked against `sionna 2.0.1` and matched bit-for-bit in testing.

Torch is optional. If it is installed and CUDA is available, the script uses it to accelerate Viterbi decoding. The NumPy path gives the same decoded bits.
