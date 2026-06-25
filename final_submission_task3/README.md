# ISIT 2026 D2I "The Still Mirror" - Task 3
### Team `the_silent_subcarriers`

Task 3 asks us to be Eve and predict the key bits Alice gets from her own noisy channel, while Eve only sees her own channel from a different transmitter position under the same RIS configuration. The metric is mean bit mismatch rate (BMR), so lower is better, and the submitted model has to stay under 40 million parameters.

The headline local mock-test result is:

```text
mean BMR = 0.1965 over the 12 public Alice/Eve pairs
noise floor = 0.099
single decode mean = 0.2223
isotropic majority vote mean = 0.2002
bootstrap majority vote mean = 0.1965
```

All numbers above can be reproduced with `the_silent_subcarriers_3.py`. Add `--baseline-ridge` to print the reference ridge baseline as well. 

## Setup

```bash
pip install -r packages.txt
```

The script expects the BRISC dataset at the path in `config.yaml`, by default:

```text
../ISIT2026-challenge-dataset
```

That folder should contain the `antenna{Dipole,Log}_pos*.mat` files and `configurations_10000.mat`.

## Constraint handling

Each condition uses one folded linear predictor:

```text
h_A_hat = design_quad(RIS) @ W_config + eve_csi @ W_eve
```

The folded predictor has `1,025,596` parameters per condition. For the 12 public conditions, that is about `12.3M` predictor parameters. The bootstrap majority vote also stores a held-out error pool as decode-time uncertainty calibration rather than as predictor weights. Even if that pool is counted conservatively, the total is `16.95M`, well under the `40M` limit.

All fitting and calibration use only training configurations from public positions. Evaluation Alice CSI is not read during fitting. Private positions are not used as auxiliary training data; they are only loaded if an evaluator explicitly asks for that condition. The majority-vote scale is selected on a train-only hold-out slice, using Alice's public 35 dB noise model.


## Run

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

The full run is CPU-bound because the majority-vote decoder runs many Viterbi passes.

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

## Extractor check

Alice's extractor is 8-level Gray quantisation followed by a rate-1/3, K=5 Viterbi decoder. One easy mistake is the Sionna LLR sign convention: the reference call `llr = (1 - 2*bit)*10` means the decoder is effectively applied to the complement of the raw Gray bits. The NumPy extractor in this package was checked against `sionna 2.0.1` and matched bit-for-bit in testing.

Torch is optional. If it is installed and CUDA is available, the script uses it to accelerate Viterbi decoding. The NumPy path gives the same decoded bits.

## The main idea

The best Eve is not just a direct CSI-transfer model. Eve knows the RIS configuration exactly, and Alice's clean CSI is mostly a deterministic function of that configuration. So the first part of the method is closer to a Task 0 model for Alice's position: learn `RIS configuration -> Alice clean CSI`.

Eve's CSI still helps, but not as the whole signal. It helps through the residual. After fitting configuration models for both Alice and Eve, we use Eve's observed residual to update Alice's predicted residual with a GLS/BLUE correction. In compact form:

```text
h_A_hat(c) = f_A(c) + Sigma_AE Sigma_EE^{-1} (h_E_obs(c) - f_E(c))
```

The implemented predictor is folded into one linear map:

```text
h_A_hat(c) = design_quad(c) @ W_config + eve_csi(c) @ W_eve
```

The feature map `design_quad(c)` contains a bias term, the 256 RIS states, and local pairwise products between nearby RIS elements. That local quadratic term is a practical way to model RIS coupling without trying to fit an enormous all-pairs model.

## Why the decoder matters

The fixed extractor is discontinuous: a small CSI error near a quantisation or Viterbi decision boundary can flip decoded bits. A single decode of `h_A_hat` is therefore too confident.

We treat the decoded bit as a posterior mode problem. Around the CSI estimate, we draw perturbed CSI samples, run Alice's extractor on each one, and take a per-bit majority vote. Two perturbation laws are calibrated on a hold-out slice of the training configurations:

- `iso`: isotropic Gaussian noise at a calibrated prediction-error scale.
- `boot`: resampled held-out error vectors from the model, plus Alice's 35 dB noise.

The bootstrap version wins on all 12 public conditions because it keeps the cross-subcarrier error structure that isotropic noise ignores.

This is the largest practical gain in the solution:

```text
single decode -> bootstrap majority vote: 0.2223 -> 0.1965 BMR
```

## Extractor correctness

A lot of the score depends on matching Alice's extractor exactly. The extractor is:

1. Stack real and imaginary CSI parts.
2. Quantise with an 8-level uniform Gray quantiser.
3. Decode with a rate-1/3, constraint-length-5 Viterbi decoder.

The reference pipeline passes `llr = (1 - 2*bit)*10` to Sionna's decoder. Sionna uses the sign convention `positive LLR -> bit 1`, so this is equivalent to decoding the complement of the raw Gray bits. The NumPy extractor in `the_silent_subcarriers_3.py` follows that behavior and was checked bit-for-bit against `sionna 2.0.1` in testing. Decoding the raw Gray bits directly gives the wrong key and produces meaningless BMR numbers.

## Model details

One independent model is fit per `(antenna, Alice position, Eve position)` condition.

```text
h_A_hat(c) = design_quad(c) @ W_config + eve_csi(c) @ W_eve
bits_hat   = MajorityVote_K(AliceExtractor(h_A_hat(c) + calibrated perturbation))
```

Parameter count per condition:

```text
W_config: 1635 x 484
W_eve:     484 x 484
total:  1,025,596 parameters
```

The majority vote is a decode-time estimator, not another learned neural network. The stored error pool is used to sample plausible prediction errors. Even if that pool is counted conservatively, the 12-condition total remains `16.95M`, below the `40M` limit.
