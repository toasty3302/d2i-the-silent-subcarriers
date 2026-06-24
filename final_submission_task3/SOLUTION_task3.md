# Task 3 solution summary
### Team `the_silent_subcarriers`

## What we are solving

Task 3 is a physical-layer security problem. Alice extracts key bits from her own noisy CSI with a fixed public extractor. Eve is at a different transmitter position. She sees her own noisy CSI under the same RIS configuration, knows the RIS state and metadata, and has to predict Alice's bits. The score is average bit mismatch rate (BMR), with a 40M-parameter limit.

Our local mock-test score is `0.1965` mean BMR over the 12 public Alice/Eve pairs. The main comparison points are:

```text
plain Eve-to-Alice CSI ridge baseline: about 0.38 BMR
configuration-only model:             about 0.27 BMR
single decode of our CSI estimate:    0.2223 BMR
isotropic majority vote:              0.2002 BMR
bootstrap majority vote:              0.1965 BMR
perfect clean-CSI noise floor:        about 0.099 BMR
```

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

## Local results

Mock-test setup: 80/20 split by RIS configuration, seed 42, both antennas, Alice positions `{1,2}`, Eve positions `{1,2,3,5}`, same-position pairs skipped.

| antenna | Alice <- Eve | boot-MV BMR | iso-MV | single | floor | CSI NMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dipole | 1 <- 2 | 0.170 | 0.175 | 0.195 | 0.092 | -24.6 dB |
| Dipole | 1 <- 3 | 0.169 | 0.173 | 0.190 | 0.093 | -24.9 dB |
| Dipole | 1 <- 5 | 0.198 | 0.201 | 0.223 | 0.091 | -21.8 dB |
| Dipole | 2 <- 1 | 0.171 | 0.173 | 0.192 | 0.111 | -26.8 dB |
| Dipole | 2 <- 3 | 0.173 | 0.177 | 0.194 | 0.104 | -26.6 dB |
| Dipole | 2 <- 5 | 0.236 | 0.242 | 0.268 | 0.101 | -19.9 dB |
| Log | 1 <- 2 | 0.205 | 0.207 | 0.233 | 0.113 | -23.0 dB |
| Log | 1 <- 3 | 0.204 | 0.206 | 0.226 | 0.116 | -23.4 dB |
| Log | 1 <- 5 | 0.255 | 0.257 | 0.291 | 0.104 | -17.0 dB |
| Log | 2 <- 1 | 0.176 | 0.179 | 0.198 | 0.087 | -23.3 dB |
| Log | 2 <- 3 | 0.174 | 0.178 | 0.195 | 0.094 | -23.9 dB |
| Log | 2 <- 5 | 0.227 | 0.232 | 0.263 | 0.083 | -16.2 dB |
| mean | | 0.1965 | 0.2002 | 0.2223 | 0.099 | |

The hardest cases are the far Eve-position pairs, especially Eve position 5. In those cases the residual sharing is weaker, so the model falls back closer to the configuration-only prediction. Those same cases are also where majority voting helps most, because the single point-estimate decode is most overconfident.

## What we tried but did not ship

The useful gains came from respecting the physical structure and from decoding better. Larger generic models did not help after the GLS residual correction.

| variant | result |
| --- | --- |
| Deep conv-net configuration model | Improved config-only NMSE, but the gain mostly disappeared after GLS; not worth the extra dependency and model files. |
| Factorisation-machine / deep MLP over RIS states | Overfit the 8k training configurations and did not improve validation behavior. |
| Bigger or longer neural nets, low-rank SVD heads | Worse or neutral. Capacity was not the limiting issue. |
| Polynomial kernel ridge, degrees 2-6 | Higher degrees became unstable or overfit; did not beat the shipped quadratic + GLS path. |
| Translation-invariant autocorrelation features | Worse, which suggests the residual coupling is position-specific rather than shift-invariant. |
| Joint nonlinear config+Eve net | Matched GLS within noise; GLS already captures the useful linear residual sharing. |
| Diagonal majority-vote perturbation | Similar to isotropic; bootstrap helped because it preserves cross-subcarrier correlations. |
| Clean-Eve oracle check | Barely improved, showing Eve's measurement noise is not the main bottleneck. |
| Direct bit classification | Worse than CSI prediction followed by the exact extractor. |

## Data use and reproducibility

All fitting uses public-position training configurations only. The train/evaluation split is by configuration ID. The majority-vote perturbation scale is selected on a train-only calibration slice; evaluation labels are not used for fitting, selection, or tuning. Private positions are only loaded if they are part of the requested evaluation condition.

The script is deterministic for a fixed seed. It needs NumPy, SciPy, h5py, and PyYAML. Torch is optional and only accelerates Viterbi decoding when CUDA is available; the NumPy and torch paths were checked to produce the same decoded bits.

Reproduce the main 12-condition run:

```bash
python the_silent_subcarriers_3.py --antenna-types Dipole Log \
       --alice-positions 1 2 --eve-positions 1 2 3 5
```
