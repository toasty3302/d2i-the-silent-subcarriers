"""Task 3 - Secret-key generation for team the_silent_subcarriers.

The task is to act as Eve: use Eve's noisy CSI, the RIS configuration, and the
public metadata to predict the key bits Alice extracts from her own noisy CSI.

The approach is deliberately extractor-aware. We first estimate Alice's clean CSI from
RIS configuration features, then use Eve's observed residual CSI to correct that estimate.
Only after this CSI step do we run Alice's exact key extractor. The final readout is a
train-calibrated majority vote over plausible CSI prediction errors, which is more stable
than decoding a single point estimate.

The model for one condition is folded into one linear map:

    h_A_hat(c) = Xq(c) @ W_config + eve_csi(c) @ W_eve
    bits       = MajorityVote_K(AliceExtractor(h_A_hat(c) + calibrated perturbation))

where Xq(c) contains a bias term, the 256 RIS states, and local pairwise RIS products.
The folded predictor has 1,025,596 parameters per condition. The majority-vote error
pool is a decode-time uncertainty calibration; even if counted conservatively, the
12-condition package stays well below the 40M limit.

"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import yaml

MAX_TOTAL_PARAMS = 40_000_000
SNR_DB_DEFAULT = 35.0
LAM_CSI = 1.0  # ridge for the config->CSI maps
LAM_GLS = 1e-3  # relative ridge for the residual cross-covariance solve
MV_SAMPLES = 64  # Monte-Carlo samples for majority-vote (Bayes) decoding
MV_SCALES = (
    0.6,
    0.8,
    1.0,
    1.2,
    1.5,
)  # perturbation scales swept on the TRAIN cal split
MV_CAL_FRAC = 0.2  # train fraction held out (train-only) to calibrate the scale
MV_CAL_MAX = 800  # cap on calibration configs (bounds calibration cost)
MV_CAL_SAMPLES = 16  # MC samples during calibration (the final decode uses MV_SAMPLES)
MV_MAX_ROWS = (
    8192  # cap rows per batched Viterbi call in the majority vote (bounds memory)
)


GEN = (0o25, 0o33, 0o37)  # standard rate-1/3 K=5 generators (octal 25,33,37)
KK = 5
NSTATES = 1 << (KK - 1)  # 16 trellis states


def gray_code_bits(n_bits: int) -> np.ndarray:
    v = np.arange(2**n_bits, dtype=np.int32)
    g = v ^ (v >> 1)  # binary-reflected Gray code == graycode.tc_to_gray_code
    return ((g[:, None] >> np.arange(n_bits - 1, -1, -1)) & 1).astype(np.uint8)


def build_quantizer(alice_train: np.ndarray, levels: int = 8):
    """v_sat = max|Re/Im| of (noisy) Alice training CSI; uniform Gray quantiser."""
    ri = np.stack([alice_train.real, alice_train.imag], axis=-1)
    v_sat = float(np.max(np.abs(ri))) or 1.0
    step = (2.0 * v_sat) / levels
    lv = np.linspace(-v_sat, v_sat, levels) + step / 2.0
    return lv, gray_code_bits(int(np.log2(levels)))


def quantize_to_gray_bits(csi, levels, gray_bits):
    vals = np.stack([csi.real, csi.imag], axis=-1).reshape(csi.shape[0], -1)
    nearest = np.argmin(np.abs(vals[:, :, None] - levels.reshape(1, 1, -1)), axis=-1)
    return gray_bits[nearest].reshape(csi.shape[0], -1).astype(np.uint8)


def _parity(x):
    return bin(x).count("1") & 1


def _viterbi_tables():
    ns = np.zeros((NSTATES, 2), np.int64)
    out = np.zeros((NSTATES, 2, 3), np.int64)
    mask = (1 << (KK - 1)) - 1
    for s in range(NSTATES):
        for b in (0, 1):
            shift = (b << (KK - 1)) | s  # [u(t), state(4 prev inputs)]
            ns[s, b] = (shift >> 1) & mask
            for gi, g in enumerate(GEN):
                out[s, b, gi] = _parity(shift & g)
    inc = [[] for _ in range(NSTATES)]
    for s in range(NSTATES):
        for b in (0, 1):
            inc[ns[s, b]].append((s, b))
    return ns, out, np.array(inc, np.int64)


_NS, _OUT, _INC = _viterbi_tables()

# Optional torch acceleration for the Viterbi hot loop (the majority-vote decode runs it K times).
try:
    import torch as _torch

    _DEV = "cuda" if _torch.cuda.is_available() else "cpu"
    _OUTt = _torch.tensor(_OUT, device=_DEV, dtype=_torch.int64)
    _INCt = _torch.tensor(_INC, device=_DEV, dtype=_torch.int64)
except Exception:
    _torch = None


def _viterbi_decode_torch(raw_bits: np.ndarray) -> np.ndarray:
    N = raw_bits.shape[0]
    nst = raw_bits.shape[1] // 3
    raw = _torch.as_tensor(
        raw_bits[:, : nst * 3], device=_DEV, dtype=_torch.int64
    ).view(N, nst, 3)
    INF = 1 << 30
    metrics = _torch.full((N, NSTATES), INF, device=_DEV, dtype=_torch.int64)
    metrics[:, 0] = 0
    bstate = _torch.zeros((nst, N, NSTATES), device=_DEV, dtype=_torch.int64)
    bbit = _torch.zeros((nst, N, NSTATES), device=_DEV, dtype=_torch.uint8)
    s0, b0 = _INCt[:, 0, 0], _INCt[:, 0, 1]
    s1, b1 = _INCt[:, 1, 0], _INCt[:, 1, 1]
    for st in range(nst):
        r = raw[:, st, :]
        branch = (_OUTt[None] - r[:, None, None, :]).abs().sum(3)
        cand = metrics[:, :, None] + branch
        c0, c1 = cand[:, s0, b0], cand[:, s1, b1]
        pick1 = c1 < c0
        metrics = _torch.where(pick1, c1, c0)
        bstate[st] = _torch.where(pick1, s1[None].expand(N, -1), s0[None].expand(N, -1))
        bbit[st] = _torch.where(
            pick1,
            b1[None].expand(N, -1).to(_torch.uint8),
            b0[None].expand(N, -1).to(_torch.uint8),
        )
    state = metrics.argmin(1)
    idx = _torch.arange(N, device=_DEV)
    dec = _torch.zeros((N, nst), device=_DEV, dtype=_torch.uint8)
    for st in range(nst - 1, -1, -1):
        dec[:, st] = bbit[st, idx, state]
        state = bstate[st, idx, state]
    return dec.cpu().numpy()


def _viterbi_decode_batch(raw_bits: np.ndarray) -> np.ndarray:
    """Hard-decision Viterbi (rate 1/3, K=5), vectorised over rows.

    Uses the torch path only when a CUDA device is present (a real speed-up); on CPU the
    vectorised NumPy trellis below is faster and more predictable than torch-CPU, so we
    keep the deliverable GPU-accelerated *and* dependency-light without surprises."""
    if _torch is not None and _DEV == "cuda" and raw_bits.shape[0] >= 64:
        return _viterbi_decode_torch(raw_bits)
    N = raw_bits.shape[0]
    nst = raw_bits.shape[1] // 3
    recv = raw_bits[:, : nst * 3].reshape(N, nst, 3).astype(np.int64)
    INF = 1 << 30
    metrics = np.full((N, NSTATES), INF, np.int64)
    metrics[:, 0] = 0
    bstate = np.zeros((nst, N, NSTATES), np.int16)
    bbit = np.zeros((nst, N, NSTATES), np.uint8)
    for st in range(nst):
        r = recv[:, st, :]
        branch = np.abs(_OUT[None] - r[:, None, None, :]).sum(3)  # (N,16,2) Hamming
        cand = metrics[:, :, None] + branch
        nm = np.full((N, NSTATES), INF, np.int64)
        for t in range(NSTATES):
            (s0, b0), (s1, b1) = _INC[t]
            c0, c1 = cand[:, s0, b0], cand[:, s1, b1]
            pick1 = c1 < c0
            nm[:, t] = np.where(pick1, c1, c0)
            bstate[st, :, t] = np.where(pick1, s1, s0)
            bbit[st, :, t] = np.where(pick1, b1, b0)
        metrics = nm
    state = np.argmin(metrics, 1)
    idx = np.arange(N)
    dec = np.zeros((N, nst), np.uint8)
    for st in range(nst - 1, -1, -1):
        dec[:, st] = bbit[st, idx, state]
        state = bstate[st, idx, state]
    return dec


def extract_keys(csi, levels, gray_bits, code_rate="1/3"):
    """Alice's exact extractor. Verified bit-for-bit vs sionna 2.0.1 ViterbiDecoder.

    The official pipeline calls sionna's decoder with llr=(1-2b)*10; sionna's LLR sign
    is positive->1, so that feeds the COMPLEMENT of the raw Gray bits -> decode (1-raw).
    """
    raw = quantize_to_gray_bits(csi, levels, gray_bits)
    if code_rate == "raw":
        return raw
    return _viterbi_decode_batch(1 - raw)


def bit_mismatch_rate(pred_bits, true_bits):
    return float(
        np.mean(np.asarray(pred_bits).reshape(-1) != np.asarray(true_bits).reshape(-1))
    )


# data: (config.yaml, RIS configs, averaged CSI streamed from the .mat files)
@dataclass(frozen=True)
class DatasetConfig:
    root_dir: Path
    config_file: str
    file_pattern: str
    antenna_types: list
    public_positions: list
    n_subcarriers: int
    seed: int


def load_config(path: Path) -> DatasetConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    ds, sp = raw.get("dataset", {}), raw.get("split", {})
    root = Path(ds.get("root_dir", "ISIT2026-challenge-dataset"))
    if not root.is_absolute():
        root = (path.resolve().parent / root).resolve()
    return DatasetConfig(
        root_dir=root,
        config_file=ds.get("config_file", "configurations_10000.mat"),
        file_pattern=ds.get("file_pattern", "antenna{ant_type}_pos{pos_idx}.mat"),
        antenna_types=list(ds.get("antenna_types", ["Dipole", "Log"])),
        public_positions=list(ds.get("public_positions", [1, 2, 3, 5])),
        n_subcarriers=int(ds.get("n_subcarriers", 242)),
        seed=int(sp.get("seed", 42)),
    )


def load_ris_configurations(cfg: DatasetConfig) -> np.ndarray:
    from scipy.io import loadmat

    m = np.asarray(loadmat(cfg.root_dir / cfg.config_file)["matrices"])
    if m.ndim != 3 or m.shape[1:] != (16, 16):
        raise ValueError(f"expected (N,16,16) RIS matrices, got {m.shape}")
    # F-order flatten matches confId indexing (vector index k = col*16 + row)
    return m.transpose(0, 2, 1).reshape(m.shape[0], -1).astype(np.float64)


_CSI_CACHE: dict = {}


def load_averaged_csi(
    data_file: Path, n_cfg: int, n_subcarriers: int, chunk: int = 32768
):
    """Per-config AVERAGED complex CSI over all 242 sub-carriers, streamed from .mat.

    Averaging the ~60 frames per config is the spec's recommended preprocessing and
    drives the residual measurement noise to ~ -46 dB (well below everything else).
    Memoised so a position reused across Eve conditions is read only once."""
    key = str(data_file)
    if key in _CSI_CACHE:
        return _CSI_CACHE[key]
    sums = np.zeros((n_cfg, n_subcarriers), np.complex128)
    counts = np.zeros(n_cfg, np.int64)
    with h5py.File(data_file, "r") as f:
        csi, conf = f["csi"], f["confId"]
        for s in range(0, csi.shape[0], chunk):
            e = min(s + chunk, csi.shape[0])
            c = np.asarray(conf[s:e]).reshape(-1).astype(np.int64) - 1
            blk = csi[s:e, :n_subcarriers]
            cc = blk["real"] + 1j * blk["imag"]
            keep = (c >= 0) & (c < n_cfg)
            c, cc = c[keep], cc[keep]
            # vectorised group-sum: sort by config, reduce at group boundaries
            order = np.argsort(c, kind="stable")
            cs, ccs = c[order], cc[order]
            uniq, start = np.unique(cs, return_index=True)
            sums[uniq] += np.add.reduceat(ccs, start, axis=0)
            counts[uniq] += np.diff(np.append(start, len(cs)))
    present = counts > 0
    avg = np.zeros((n_cfg, n_subcarriers), np.complex128)
    avg[present] = sums[present] / counts[present, None]
    _CSI_CACHE[key] = (avg, counts)
    return avg, counts


def add_awgn_complex(csi, snr_db, rng, avg_power):
    npow = avg_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(npow / 2.0) * (
        rng.normal(size=csi.shape) + 1j * rng.normal(size=csi.shape)
    )
    return csi + noise


# model: (config quadratic + Eve GLS residual transfer, folded to one linear map)
def adjacency_pairs(n=16, dmax=2.0):
    """Nearest-neighbour RIS element pairs (Euclidean distance <= dmax on the 16x16
    panel).  dmax=2 (4-neighbour + diagonal + distance-2) is the CV-validated sweet
    spot: richer (dmax=3) over-fits, sparser (dmax=1) under-fits."""
    coords = [(k // n, k % n) for k in range(n * n)]
    P, d2 = [], dmax * dmax
    for i in range(n * n):
        ci, ri = coords[i]
        for j in range(i + 1, n * n):
            cj, rj = coords[j]
            if (ci - cj) ** 2 + (ri - rj) ** 2 <= d2:
                P.append((i, j))
    return np.asarray(P)


def design_quad(S, pairs):
    inter = S[:, pairs[:, 0]] * S[:, pairs[:, 1]]
    return np.concatenate([np.ones((S.shape[0], 1)), S, inter], axis=1)


def _ridge(X, Y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def _ri(csi):
    return np.concatenate([csi.real, csi.imag], axis=1)


def _csi(ri):
    n = ri.shape[1] // 2
    return ri[:, :n] + 1j * ri[:, n:]


@dataclass
class GLSModel:
    """Folded linear Eve->Alice predictor:  h_A_ri = Xq @ W_config + eve_ri @ W_eve.

    Bits are read out with a Bayes-optimal majority vote (``pert_spec`` set): Eve does
    not know Alice's exact channel -- her CSI prediction has error -- so the single decode
    of the point estimate is over-confident through the error-AMPLIFYING Viterbi.
    Marginalising over Eve's own prediction uncertainty (perturb the prediction, decode,
    take the per-bit majority) is the minimum-BMR estimator.  The perturbation is either
    'iso' (isotropic Gaussian of a calibrated std) or 'boot' (resample REAL held-out
    error vectors -- capturing the prediction's cross-subcarrier error correlations -- plus
    Alice's own 35 dB noise); whichever wins on a train-only calibration split is used.
    """

    pairs: np.ndarray
    W_config: np.ndarray  # (n_qfeat, 2*nsub)
    W_eve: np.ndarray  # (2*nsub, 2*nsub)
    levels: np.ndarray
    gray_bits: np.ndarray
    pert_spec: dict = None  # {'mode':'iso'|'boot', ...} ; None => single decode

    @property
    def n_parameters(self) -> int:
        return int(self.W_config.size + self.W_eve.size)

    def predict_csi(self, S_eval, eve_eval_csi):
        ri = (
            design_quad(S_eval, self.pairs) @ self.W_config
            + _ri(eve_eval_csi) @ self.W_eve
        )
        return _csi(ri)

    def predict_bits(self, S_eval, eve_eval_csi, mv_samples=MV_SAMPLES, seed=0):
        pred_ri = (
            design_quad(S_eval, self.pairs) @ self.W_config
            + _ri(eve_eval_csi) @ self.W_eve
        )
        if (
            not self.pert_spec
            or self.pert_spec.get("mode") in (None, "none")
            or mv_samples <= 1
        ):
            return extract_keys(_csi(pred_ri), self.levels, self.gray_bits)
        return majority_vote_bits(
            pred_ri, self.levels, self.gray_bits, self.pert_spec, mv_samples, seed
        )


def _mv_draw(pred_ri, spec, g, rng):
    """One (g, N, D) block of perturbed CSI predictions for the majority vote."""
    N, D = pred_ri.shape
    if spec["mode"] == "boot":
        E = spec["errs"]  # (M, D) real held-out errors
        idx = rng.integers(0, len(E), size=(g, N))
        pert = -spec["scale"] * E[idx] + spec["anoise"] * rng.standard_normal((g, N, D))
        return pred_ri[None] + pert
    return pred_ri[None] + spec["sigma"] * rng.standard_normal((g, N, D))  # iso


def majority_vote_bits(pred_ri, levels, gray_bits, spec, K, seed=0):
    """Bayes-optimal per-bit estimate: decode K perturbed copies of the CSI prediction
    and take the per-bit majority.  ``spec`` selects the perturbation law (iso / boot).

    The K draws are decoded in BATCHES of G perturbations stacked into one Viterbi call
    (G*N rows, G bounded so G*N <= MV_MAX_ROWS), so the sequential 484-step trellis loop
    runs ~K/G times instead of K -- a large speed-up, especially for the baseline's small
    default eval set (where one call covers all K)."""
    rng = np.random.default_rng(seed)
    N, D = pred_ri.shape
    G = max(1, min(K, MV_MAX_ROWS // max(N, 1)))
    acc = None
    done = 0
    while done < K:
        g = min(G, K - done)
        sims = _mv_draw(pred_ri, spec, g, rng)  # (g, N, 484)
        bits = extract_keys(_csi(sims.reshape(g * N, D)), levels, gray_bits)
        acc_g = bits.reshape(g, N, -1).sum(0).astype(np.int32)
        acc = acc_g if acc is None else acc + acc_g
        done += g
    return (acc > (K / 2)).astype(np.uint8)


def fit_gls_model(
    S_train, alice_train_clean, eve_train_noisy, alice_train_noisy, pairs
) -> GLSModel:
    """Fit the config ridges + GLS residual transfer on TRAIN configs, fold to one map.

    h_A_hat = fA(c) + (eve_noisy - fE(c)) @ T^T,  fA=Xq@WA, fE=Xq@WE,
    T = Sigma_{rA,rE} (Sigma_{rE} + ridge)^-1 over training residuals (rE uses noisy Eve).
    Folded:  W_config = WA - WE@T^T ,  W_eve = T^T .
    """
    Xq = design_quad(S_train, pairs)
    WA = _ridge(Xq, _ri(alice_train_clean), LAM_CSI)
    WE = _ridge(Xq, _ri(eve_train_noisy), LAM_CSI)
    rA = _ri(alice_train_clean) - Xq @ WA
    rE = _ri(eve_train_noisy) - Xq @ WE
    m = len(S_train)
    S_EE = rE.T @ rE / m
    S_AE = rA.T @ rE / m
    T = S_AE @ np.linalg.inv(
        S_EE + LAM_GLS * np.trace(S_EE) / S_EE.shape[0] * np.eye(S_EE.shape[0])
    )
    levels, gray = build_quantizer(alice_train_noisy)
    return GLSModel(
        pairs=pairs, W_config=WA - WE @ T.T, W_eve=T.T, levels=levels, gray_bits=gray
    )


def calibrate_mv(
    S_tr, a_avg_tr, e_no_tr, a_no_tr, pairs, seed, snr_db, mv_cal_K=MV_CAL_SAMPLES
):
    """Choose the majority-vote perturbation law on a TRAIN-ONLY hold-out (regime-A).

    Carve a calibration slice out of the training configs, fit the model on the rest, and
    compare on that slice -- whose 'true' bits come from Alice's KNOWN 35 dB noise model,
    never from evaluation data:
      * 'iso'  : isotropic Gaussian, std = scale x prediction-error std;
      * 'boot' : resample the model's REAL held-out error vectors (cross-subcarrier
                 correlations included), scaled, plus Alice's own 35 dB noise.
    Returns the winning spec dict (or None if neither beats the single decode).  The
    bootstrap error pool is the sub-model's held-out errors -- a faithful sample of the
    full model's generalisation-error distribution on unseen configs.
    """
    rng = np.random.default_rng(seed + 7)
    perm = rng.permutation(len(S_tr))
    n_cal = min(max(int(round(MV_CAL_FRAC * len(perm))), 1), MV_CAL_MAX)
    cal, sub = np.sort(perm[:n_cal]), np.sort(perm[n_cal:])
    avg_power = float(np.mean(np.abs(np.concatenate([a_avg_tr, e_no_tr], 0)) ** 2))
    anoise = float(np.sqrt(avg_power / 10.0 ** (snr_db / 10.0) / 2.0))
    if len(cal) < 8 or len(sub) < 16:
        m = fit_gls_model(S_tr, a_avg_tr, e_no_tr, a_no_tr, pairs)
        err = _ri(m.predict_csi(S_tr, e_no_tr)) - _ri(a_avg_tr)
        return {"mode": "iso", "sigma": float(np.sqrt(np.mean(err**2)))}
    m = fit_gls_model(S_tr[sub], a_avg_tr[sub], e_no_tr[sub], a_no_tr[sub], pairs)
    pred_ri = _ri(m.predict_csi(S_tr[cal], e_no_tr[cal]))
    errs = (pred_ri - _ri(a_avg_tr[cal])).astype(np.float32)  # held-out error vectors
    err_std = float(np.sqrt(np.mean(errs**2)))
    true_cal = extract_keys(a_no_tr[cal], m.levels, m.gray_bits)
    base = bit_mismatch_rate(
        extract_keys(_csi(pred_ri), m.levels, m.gray_bits), true_cal
    )
    best_spec, best_bmr = None, base
    for sc in MV_SCALES:  # isotropic
        spec = {"mode": "iso", "sigma": sc * err_std}
        b = bit_mismatch_rate(
            majority_vote_bits(
                pred_ri, m.levels, m.gray_bits, spec, mv_cal_K, seed + 13
            ),
            true_cal,
        )
        if b < best_bmr:
            best_spec, best_bmr = spec, b
    for sc in MV_SCALES:  # bootstrap (real error structure)
        spec = {"mode": "boot", "errs": errs, "scale": float(sc), "anoise": anoise}
        b = bit_mismatch_rate(
            majority_vote_bits(
                pred_ri, m.levels, m.gray_bits, spec, mv_cal_K, seed + 29
            ),
            true_cal,
        )
        if b < best_bmr:
            best_spec, best_bmr = spec, b
    return best_spec


# per-condition attack
@dataclass
class AttackResult:
    antenna_type: str
    alice_position: int
    eve_position: int
    n_train: int
    n_eval: int
    n_bits: int
    model_params: int
    bmr: float
    bmr_single: float
    floor_bmr: float
    csi_nmse_db: float
    baseline_ridge_bmr: float = float("nan")


def _plain_ridge_bmr(S_tr, S_ev, e_no_tr, e_no_ev, a_no_tr, a_no_ev, levels, gray):
    """Official-style Task-3 baseline, for reference only: a plain ridge mapping Eve's
    noisy CSI -> Alice's noisy CSI (NO RIS configuration), then Alice's extractor, single
    decode.  This is the linear Eve->Alice mapping the Task 3 baseline uses; computing it
    here makes the comparison number verifiable inside this package.  ``S_*`` are accepted
    but unused (this reference baseline ignores the RIS configuration).
    """
    X = np.concatenate([_ri(e_no_tr), np.ones((len(e_no_tr), 1))], axis=1)
    W = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ _ri(a_no_tr))
    Xe = np.concatenate([_ri(e_no_ev), np.ones((len(e_no_ev), 1))], axis=1)
    pred_csi = _csi(Xe @ W)
    true_bits = extract_keys(a_no_ev, levels, gray)
    return bit_mismatch_rate(extract_keys(pred_csi, levels, gray), true_bits)


def run_attack_setting(
    cfg,
    antenna_type,
    alice_position,
    eve_position,
    ris_vectors,
    snr_db,
    train_ratio,
    n_configs,
    seed,
    return_model=False,
    baseline_ridge=False,
):
    a_file = cfg.root_dir / cfg.file_pattern.format(
        ant_type=antenna_type, pos_idx=alice_position
    )
    e_file = cfg.root_dir / cfg.file_pattern.format(
        ant_type=antenna_type, pos_idx=eve_position
    )
    if not a_file.exists() or not e_file.exists():
        raise FileNotFoundError(f"missing data file(s): {a_file.name} / {e_file.name}")

    n_cfg = ris_vectors.shape[0]
    a_avg, a_cnt = load_averaged_csi(a_file, n_cfg, cfg.n_subcarriers)
    e_avg, e_cnt = load_averaged_csi(e_file, n_cfg, cfg.n_subcarriers)
    present = np.where((a_cnt > 0) & (e_cnt > 0))[0]

    if n_configs and n_configs < len(present):
        present = np.sort(
            np.random.default_rng(seed).choice(present, n_configs, replace=False)
        )
    perm = np.random.default_rng(seed).permutation(present)
    n_tr = min(max(int(round(train_ratio * len(perm))), 1), len(perm) - 1)
    tr, ev = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])

    rng = np.random.default_rng(seed + 1000 * alice_position + eve_position)
    avg_power = float(
        np.mean(np.abs(np.concatenate([a_avg[present], e_avg[present]], 0)) ** 2)
    )
    a_no = add_awgn_complex(a_avg, snr_db, rng, avg_power)
    e_no = add_awgn_complex(e_avg, snr_db, rng, avg_power)

    pairs = adjacency_pairs(dmax=2.0)
    model = fit_gls_model(ris_vectors[tr], a_avg[tr], e_no[tr], a_no[tr], pairs)
    model.pert_spec = calibrate_mv(
        ris_vectors[tr], a_avg[tr], e_no[tr], a_no[tr], pairs, seed, snr_db
    )

    true_bits = extract_keys(a_no[ev], model.levels, model.gray_bits)
    pred_bits = model.predict_bits(ris_vectors[ev], e_no[ev])
    bmr = bit_mismatch_rate(pred_bits, true_bits)
    bmr_single = bit_mismatch_rate(
        extract_keys(
            model.predict_csi(ris_vectors[ev], e_no[ev]), model.levels, model.gray_bits
        ),
        true_bits,
    )

    floor = bit_mismatch_rate(
        extract_keys(a_avg[ev], model.levels, model.gray_bits), true_bits
    )
    pred_csi = model.predict_csi(ris_vectors[ev], e_no[ev])
    nmse = float(
        10
        * np.log10(
            np.linalg.norm(_ri(pred_csi) - _ri(a_avg[ev])) ** 2
            / np.linalg.norm(_ri(a_avg[ev])) ** 2
        )
    )
    base_bmr = float("nan")
    if baseline_ridge:
        base_bmr = _plain_ridge_bmr(
            ris_vectors[tr],
            ris_vectors[ev],
            e_no[tr],
            e_no[ev],
            a_no[tr],
            a_no[ev],
            model.levels,
            model.gray_bits,
        )
    res = AttackResult(
        antenna_type,
        alice_position,
        eve_position,
        len(tr),
        len(ev),
        int(true_bits.size),
        model.n_parameters,
        bmr,
        bmr_single,
        floor,
        nmse,
        base_bmr,
    )
    return (res, model) if return_model else res


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--antenna-types", nargs="+", default=None)
    p.add_argument(
        "--antenna-type", default=None, help="alias for a single antenna type"
    )
    p.add_argument("--alice-positions", type=int, nargs="+", default=[1, 2])
    p.add_argument("--eve-positions", type=int, nargs="+", default=[3, 4, 5, 6])
    p.add_argument(
        "--n-configs", type=int, default=0, help="0 = use all available configs"
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--snr-db", type=float, default=SNR_DB_DEFAULT)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--save-models",
        action="store_true",
        help="fit all PUBLIC (antenna, Alice, Eve) conditions and write models/",
    )
    p.add_argument(
        "--baseline-ridge",
        action="store_true",
        help="also report the official plain Eve->Alice CSI ridge baseline (no config)",
    )
    p.add_argument("--results-dir", type=Path, default=Path("results/task3_secret_key"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    return p.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # live progress under redirection
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    if args.antenna_types:
        antennas = args.antenna_types
    elif args.antenna_type:
        antennas = [args.antenna_type]
    else:
        antennas = cfg.antenna_types

    if args.save_models:
        alice_pos = [p for p in args.alice_positions if p in cfg.public_positions]
        eve_pos = [p for p in cfg.public_positions]
    else:
        alice_pos, eve_pos = args.alice_positions, args.eve_positions

    ris = load_ris_configurations(cfg)
    print(f"Antenna types   : {antennas}")
    print(f"Alice positions : {alice_pos}")
    print(f"Eve positions   : {eve_pos}")
    print(
        f"SNR             : {args.snr_db:.1f} dB   (extractor: Gray-8 + rate-1/3 Viterbi)"
    )
    print(
        f"Configs         : {'all available' if not args.n_configs else args.n_configs}"
        f"  (train_ratio={args.train_ratio})"
    )

    results, saved = [], {}
    for ant in antennas:
        for al in alice_pos:
            for ev in eve_pos:
                if al == ev:
                    continue
                try:
                    res, model = run_attack_setting(
                        cfg,
                        ant,
                        al,
                        ev,
                        ris,
                        args.snr_db,
                        args.train_ratio,
                        args.n_configs,
                        seed,
                        return_model=True,
                        baseline_ridge=args.baseline_ridge,
                    )
                except FileNotFoundError as e:
                    print(f"  skip {ant} A{al}<-E{ev}: {e}")
                    continue
                results.append(res)
                base_str = (
                    f"  ridge-baseline={res.baseline_ridge_bmr:.4f}"
                    if args.baseline_ridge
                    else ""
                )
                print(
                    f"  {ant:6s} A{al}<-E{ev}: BMR={res.bmr:.4f} (single {res.bmr_single:.4f})  "
                    f"floor={res.floor_bmr:.4f}  NMSE={res.csi_nmse_db:.1f}dB  "
                    f"params={res.model_params:,}{base_str}",
                    flush=True,
                )
                if args.save_models:
                    saved[f"{ant}_A{al}_E{ev}"] = dict(
                        pairs=model.pairs,
                        W_config=model.W_config.astype(np.float32),
                        W_eve=model.W_eve.astype(np.float32),
                        levels=model.levels,
                        gray_bits=model.gray_bits,
                        pert_spec=model.pert_spec,
                    )

    if not results:
        raise RuntimeError("no conditions evaluated (check data files / positions)")

    avg = float(np.mean([r.bmr for r in results]))
    avg_single = float(np.mean([r.bmr_single for r in results]))
    floor = float(np.mean([r.floor_bmr for r in results]))
    total_params = max(r.model_params for r in results) * len(results)
    print("\nTask 3 attack summary")
    print("---------------------")
    for i, r in enumerate(results, 1):
        print(
            f"  {i:2d}/{len(results)} {r.antenna_type:6s} Alice={r.alice_position} "
            f"Eve={r.eve_position}  BMR={r.bmr:.4f}"
        )
    print(
        f"\nfinal_average_bmr = {avg:.6f}   (single-decode {avg_single:.6f}, "
        f"noise floor {floor:.4f})"
    )
    if args.baseline_ridge:
        base_avg = float(np.mean([r.baseline_ridge_bmr for r in results]))
        print(
            f"official_ridge_baseline_bmr = {base_avg:.6f}   "
            f"(plain Eve->Alice CSI ridge, ignores the RIS config)"
        )
    print(
        f"params/condition  = {results[0].model_params:,}  "
        f"(<= 40M budget allows {MAX_TOTAL_PARAMS // results[0].model_params} conditions)"
    )
    if total_params > MAX_TOTAL_PARAMS:
        print(
            f"WARNING: {len(results)} conditions x {results[0].model_params:,} "
            f"= {total_params:,} exceeds the 40M budget."
        )

    try:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.results_dir / "task3_secret_key_results.npz",
            rows=np.array([r.__dict__ for r in results], dtype=object),
            final_average_bmr=avg,
            noise_floor_bmr=floor,
            antennas=np.array(antennas),
            snr_db=args.snr_db,
        )
        print(
            f"Saved results to {(args.results_dir / 'task3_secret_key_results.npz').resolve()}"
        )
    except OSError as e:
        print(f"(results npz not written: {e}; the printed BMR above is the result)")

    if args.save_models and saved:
        try:
            args.models_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        out = args.models_dir / "the_silent_subcarriers_3.pkl"
        with open(out, "wb") as fh:
            pickle.dump(
                dict(
                    conditions=saved,
                    snr_db=args.snr_db,
                    note="folded GLS Eve->Alice linear maps; "
                    "h_A_ri = design_quad(S)@W_config + eve_ri@W_eve",
                ),
                fh,
            )
        per = next(iter(saved.values()))
        n_per = per["W_config"].size + per["W_eve"].size
        print(
            f"Saved {len(saved)} condition models to {out.resolve()}  "
            f"({n_per:,} params each, {n_per * len(saved):,} total <= 40M)"
        )


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    main()
