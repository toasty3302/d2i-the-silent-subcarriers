"""
Team `the_silent_subcarriers` --- ISIT 2026 D2I "The Still Mirror", Task 0.
=========================================================================

Task 0 -- given a 16x16 binary RIS configuration, predict the complex channel
(CSI): 242 sub-carriers => 484 stacked real/imag numbers, for each of the
8 measurement conditions = {Dipole, Log} antenna x positions {1, 2, 3, 5}.

We train ONE independent predictor per condition (8 in total). Each predictor is
a blend of two models that make *different* errors:

  prediction = convex_blend( quad , cnn_ensemble )                       (per SVD component)

  * quad : a closed-form ridge regression on physically motivated features
           [bias, 256 RIS reflection bits, 1808 nearest-neighbour coupling
           products  s_i * s_j], solved in a fixed rank-32 SVD subspace of the
           channel. No iterative training -- a single linear solve.
  * cnn  : a small CNN ensemble. Each net: a learnable per-element complex
           reflection lift -> conv stem -> residual conv blocks (squeeze-excite)
           -> (Log only) one self-attention layer over the 256 panel positions
           -> MLP, decoded through the same rank-32 SVD basis.
           Log: 4 seeds + attention.  Dipole: 2 seeds, no attention.

On top of that, a leak-free **config-structure refinement**: the public config
set contains highly structured "macro-block" layouts (config_id 1..16 are four
uniform 8x8 blocks; config_id 17..528 are nine uniform blocks on a (5,5,6) grid).
For those test configs the channel is an exact affine/quadratic function of just
4 or 9 macro-bits, so we replace the CNN's prediction there with a tiny
closed-form macro-block model fit on the training configs of the same family.

================================  RULES  ================================
* Regime-A (no leakage). Every model is fit ONLY on the public training configs
  (train.csv: RIS bits + measured CSI). The 2000 held-out *test* configs are used
  for INPUT (their RIS bits) only -- their channels are never read during fit or
  model selection. The private positions {4,6,7,8,9} are never read at all.
* Parameter budget. Each of the 8 predictors has < 1,000,000 trainable params:
      Log    : 4 x 223,308 (cnn) + 82,052 (quad) =   975,284  <= 1,000,000
      Dipole : 2 x 412,600 (cnn) + 82,052 (quad) =   907,252  <= 1,000,000
  Run `python the_silent_subcarriers_0.py params` to print/verify these.

================================  USAGE  ================================
  # 0. one-off: turn the official Kaggle CSVs into the per-condition .npy
  #    arrays this script reads (also writes train/test config_id arrays).
  python prep_data.py  <kaggle_dir>  <data_dir>          # e.g. kaggle_data  data

  # 1. reproduce the submission from the shipped weights (CPU, ~1-2 min, no GPU):
  DATA=<data_dir> python the_silent_subcarriers_0.py reproduce
        -> writes submission.csv

  # 2. (optional) retrain all 8 predictors from scratch (GPU recommended), then
  #    rebuild the submission with the freshly trained weights:
  DATA=<data_dir> python the_silent_subcarriers_0.py train
        -> writes models/the_silent_subcarriers_0.pth  and  submission.csv

  # 3. print the per-predictor parameter counts (budget compliance):
  python the_silent_subcarriers_0.py params

`<data_dir>` must contain, per condition c in CONDS:
    train_{c}_ris.npy  (N x 256 int8 RIS bits)   train_{c}_csi.npy (N x 484 f32 CSI)
    train_{c}_cfg.npy  (N int   config_id)
    test_{c}_ris.npy   (M x 256 int8)            test_{c}_eid.npy  (M object example_id)
    test_{c}_cfg.npy   (M int   config_id)
"""
import sys, os, csv, numpy as np

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
CONDS = ["Dipole_pos1", "Dipole_pos2", "Dipole_pos3", "Dipole_pos5",
         "Log_pos1", "Log_pos2", "Log_pos3", "Log_pos5"]
K = 32                                   # SVD rank of the per-condition output head
MODEL_PATH = "models/the_silent_subcarriers_0.pth"
DATA = os.environ.get("DATA", "data")
OFF_REFLECT = -0.4                       # reflection magnitude of an OFF element (ON = +1.0)

# Per-condition CNN architecture (used both to TRAIN and to count params). All
# kept small so that  seeds * one_net + quad  stays under the 1M budget.
CFG = {
    "Log":    dict(seeds=4, ch=40, blocks=3, mh=48,  attn=1, epochs=350, alpha=0.15),
    "Dipole": dict(seeds=2, ch=40, blocks=2, mh=160, attn=0, epochs=250, alpha=0.25),
}
def cfg_for(c): return CFG["Log" if c.startswith("Log") else "Dipole"]

# ----------------------------------------------------------------------------
# closed-form quadratic model (physics features in a rank-K SVD subspace)
# ----------------------------------------------------------------------------
# The channel is approximately affine in the per-element reflection plus a sum of
# pairwise couplings s_i * s_j. We keep only the 1808 nearest-neighbour pairs
# (by Manhattan distance on the 16x16 panel) -- the only couplings the 8000
# training configs can actually identify -- giving 1+256+1808 features.
_CO   = [(i // 16, i % 16) for i in range(256)]
_ALLP = np.array([(i, j) for i in range(256) for j in range(i + 1, 256)])
_DIST = np.array([abs(_CO[i][0] - _CO[j][0]) + abs(_CO[i][1] - _CO[j][1]) for i, j in _ALLP])
PAIRS = _ALLP[np.argsort(_DIST, kind="stable")[:1808]]
def qfeat(Spm):                          # Spm: RIS bits already mapped to +-1
    return np.hstack([np.ones((len(Spm), 1)), Spm, Spm[:, PAIRS[:, 0]] * Spm[:, PAIRS[:, 1]]])
def quad_pred(S, C, St, B, m):
    """Ridge-fit the quad features to the rank-K SVD coeffs of (train CSI - mean),
    then predict the test configs. S/St are {0,1} RIS bit matrices."""
    X, Xt = qfeat(S * 2 - 1), qfeat(St * 2 - 1)
    A = X.T @ X
    W = np.linalg.solve(A + 100.0 * np.eye(A.shape[0]), X.T @ ((C - m) @ B))
    return m + (Xt @ W) @ B.T

def svd_basis(C):
    """Rank-K SVD subspace of the training channel (mean + top-K right singular vecs)."""
    m = C.mean(0).astype(np.float64)
    _, _, Vt = np.linalg.svd((C.astype(np.float64) - m), full_matrices=False)
    return B_from(Vt), m
def B_from(Vt): return Vt[:K].T

# ----------------------------------------------------------------------------
# config-structure refinement (leak-free: fit on TRAIN configs of each family)
# ----------------------------------------------------------------------------
# Macro-block layouts. Element n sits at row = n % 16, col = n // 16 (column-major /
# F-order flattening). config_id 1..16  => four uniform 8x8 blocks (2 bits each axis).
#                       config_id 17..528 => nine uniform blocks on a (5,5,6) grid.
def mblock9(n):
    row, col = n % 16, n // 16
    mr = 0 if row < 5 else (1 if row < 10 else 2)
    mc = 0 if col < 5 else (1 if col < 10 else 2)
    return mr * 3 + mc
def mblock4(n):
    row, col = n % 16, n // 16
    return (0 if row < 8 else 1) * 2 + (0 if col < 8 else 1)
MB9 = np.array([mblock9(n) for n in range(256)])
MB4 = np.array([mblock4(n) for n in range(256)])

def _mfeat(S, MB, nblk, quad):
    """Per-config macro-block reflection features [bias, nblk macro-bits, (pairwise)]."""
    ms = np.stack([S[:, MB == b][:, 0] for b in range(nblk)], 1)        # one bit per macro-block
    r = np.where(ms > 0.5, 1.0, OFF_REFLECT)
    F = [np.ones((len(r), 1)), r]
    if quad:
        F.append(np.stack([r[:, i] * r[:, j] for i in range(nblk) for j in range(i + 1, nblk)], 1))
    return np.hstack(F)

def structure_replace(pred, S, C, cfg, St, tcfg, teid):
    """Overwrite the structured test configs in `pred` (dict eid->484 vec) with a tiny
    closed-form macro-block model. Fit per family on the TRAIN configs only."""
    B, m = svd_basis(C)
    n = 0
    for lo, hi, MB, nblk, quad in [(17, 528, MB9, 9, True), (1, 16, MB4, 4, False)]:
        trm, tem = (cfg >= lo) & (cfg <= hi), (tcfg >= lo) & (tcfg <= hi)
        if tem.sum() == 0:
            continue
        Xtr = _mfeat(S[trm], MB, nblk, quad)
        W = np.linalg.solve(Xtr.T @ Xtr + 1e-2 * np.eye(Xtr.shape[1]), Xtr.T @ ((C[trm] - m) @ B))
        out = m + (_mfeat(St[tem], MB, nblk, quad) @ W) @ B.T
        for e, p in zip(teid[tem], out):
            pred[str(e)] = p.astype(np.float32)
        n += int(tem.sum())
    return n

# ----------------------------------------------------------------------------
# CNN (only imported/used in `reproduce` and `train`; numpy paths need no torch)
# ----------------------------------------------------------------------------
def _torch():
    import torch, torch.nn as nn, torch.nn.functional as F
    class Lift(nn.Module):                                   # learnable per-element complex reflection
        def __init__(s, n=256):
            super().__init__()
            s.a0 = nn.Parameter(torch.full((n,), 0.4)); s.a1 = nn.Parameter(torch.full((n,), 1.0))
            s.p0 = nn.Parameter(torch.full((n,), float(np.pi)) + 0.01 * torch.randn(n))
            s.p1 = nn.Parameter(0.01 * torch.randn(n))
        def forward(s, r):
            a = s.a0 + (s.a1 - s.a0) * r; p = s.p0 + (s.p1 - s.p0) * r
            return torch.stack([a * torch.cos(p), a * torch.sin(p)], 1)
    class ResBlock(nn.Module):
        def __init__(s, ch):
            super().__init__(); s.c1 = nn.Conv2d(ch, ch, 3, padding=1); s.c2 = nn.Conv2d(ch, ch, 3, padding=1)
            s.se = nn.Sequential(nn.Linear(ch, ch // 2), nn.GELU(), nn.Linear(ch // 2, ch), nn.Sigmoid())
        def forward(s, x):
            h = F.gelu(s.c1(x)); h = s.c2(h)
            g = s.se(h.mean((2, 3))).unsqueeze(-1).unsqueeze(-1); return F.gelu(x + h * g)
    class AttnBlock(nn.Module):                              # global pairwise coupling a conv cannot see
        def __init__(s, ch, heads=4):
            super().__init__(); s.norm = nn.LayerNorm(ch); s.att = nn.MultiheadAttention(ch, heads, batch_first=True)
            s.ff = nn.Sequential(nn.LayerNorm(ch), nn.Linear(ch, 2 * ch), nn.GELU(), nn.Linear(2 * ch, ch))
        def forward(s, x):
            N, ch, H, Wd = x.shape; t = x.flatten(2).transpose(1, 2)
            t = t + s.att(s.norm(t), s.norm(t), s.norm(t))[0]; t = t + s.ff(t)
            return t.transpose(1, 2).reshape(N, ch, H, Wd)
    class BigNet(nn.Module):
        def __init__(s, B, m, ch, blocks, mh, attn):
            super().__init__(); s.lift = Lift(); s.stem = nn.Conv2d(2, ch, 3, padding=1)
            s.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
            s.attn = nn.Sequential(*[AttnBlock(ch) for _ in range(attn)]) if attn else nn.Identity()
            s.head = nn.Conv2d(ch, 8, 1); s.bil = nn.Linear(512, K)
            s.mlp = nn.Sequential(nn.Linear(2048, mh), nn.GELU(), nn.Linear(mh, K))
            s.register_buffer("B", torch.tensor(B)); s.register_buffer("mm", torch.tensor(m))
        def forward(s, r):
            L = s.lift(r); x = F.gelu(s.stem(L.view(-1, 2, 16, 16))); x = s.blocks(x); x = s.attn(x)
            f = F.gelu(s.head(x)).reshape(L.size(0), -1)
            return s.mm + (s.bil(L.reshape(L.size(0), -1)) + s.mlp(f)) @ s.B.T
    return torch, nn, F, Lift, ResBlock, AttnBlock, BigNet

def cnn_params(cf):
    """Trainable params of ONE BigNet for a given config (for the budget check)."""
    torch, nn, F, Lift, ResBlock, AttnBlock, BigNet = _torch()
    net = BigNet(np.zeros((484, K), np.float32), np.zeros(484, np.float32),
                 cf["ch"], cf["blocks"], cf["mh"], cf["attn"])
    return sum(p.numel() for p in net.parameters())

QUAD_PARAMS = (1 + 256 + 1808) * K + 484 * K + 484          # Wq + SVD basis + mean
# macro-block refinement head (see structure_replace): MB9 quad (1+9+36 feats) + MB4
# linear (1+4 feats), each mapped to the same K-dim SVD output as the quad above -- the
# 484*K SVD basis is shared (NOT double-counted). Counted toward the per-predictor budget.
MBLOCK_PARAMS = (1 + 9 + 36) * K + (1 + 4) * K

# ----------------------------------------------------------------------------
# data loading (regime-A: train CSI + test RIS only; NO test channels anywhere)
# ----------------------------------------------------------------------------
def load(c):
    g = lambda s: np.load(f"{DATA}/{s}")
    S  = g(f"train_{c}_ris.npy").astype(np.float32)
    C  = g(f"train_{c}_csi.npy").astype(np.float64)
    cfg = g(f"train_{c}_cfg.npy")
    St  = g(f"test_{c}_ris.npy").astype(np.float32)
    tcfg = g(f"test_{c}_cfg.npy")
    teid = np.load(f"{DATA}/test_{c}_eid.npy", allow_pickle=True)  # object array of example_ids
    return S, C, cfg, St, tcfg, teid

# ----------------------------------------------------------------------------
# blend + submission writing
# ----------------------------------------------------------------------------
def apply_blend(c, cnn, q, m, B, blend):
    """Combine the CNN ensemble and the quad. Per-component convex blend a_k (fit on
    held-out TRAIN OOF) when present; else the scalar fallback alpha."""
    bw = blend.get(c, {}) if blend else {}
    if bw.get("mode") == "convex":
        a = np.asarray(bw["a"], np.float64); cnk = (cnn - m) @ B; qk = (q - m) @ B
        coeff = np.stack([a[k] * qk[:, k] + (1 - a[k]) * cnk[:, k] for k in range(K)], 1)
        return m + coeff @ B.T
    a = cfg_for(c)["alpha"]                                  # scalar fallback
    return a * q + (1 - a) * cnn

COLS = (["example_id"] + [f"csi_real_{i:03d}" for i in range(242)]
        + [f"csi_imag_{i:03d}" for i in range(242)])
def write_csv(path, pred_by_eid, order):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(COLS)
        for e in order:
            w.writerow([e] + [f"{x:.6g}" for x in pred_by_eid[e]])
    print(f"wrote {path}  ({len(order):,} rows)")

# ----------------------------------------------------------------------------
# reproduce: saved weights + data  ->  submission.csv   (CPU, no GPU)
# ----------------------------------------------------------------------------
def reproduce():
    import torch
    torch, nn, F, Lift, ResBlock, AttnBlock, BigNet = _torch()
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    models, blend = ckpt["models"], ckpt.get("blend", {})
    def build(sd):                                           # rebuild a BigNet from its state_dict
        ch = sd["stem.weight"].shape[0]; mh = sd["mlp.0.weight"].shape[0]
        nb = len([k for k in sd if k.startswith("blocks.") and k.endswith(".c1.weight")])
        at = len([k for k in sd if ".att.in_proj_weight" in k])
        net = BigNet(sd["B"].numpy(), sd["mm"].numpy(), ch, nb, mh, at)
        net.load_state_dict(sd); net.eval(); return net
    pred, order = {}, []
    for c in CONDS:
        S, C, cfg, St, tcfg, teid = load(c)
        Stt = torch.tensor(St)
        ps = []
        for sd in models[c]["cnn_seeds"]:
            net = build(sd)
            with torch.no_grad():
                ps.append(net(Stt).numpy().astype(np.float64))
        cnn = np.mean(ps, 0)
        B, m = svd_basis(C)
        q = quad_pred(S, C, St, B, m)
        base = apply_blend(c, cnn, q, m, B, blend)
        for e, p in zip(teid, base):
            pred[str(e)] = p.astype(np.float32); order.append(str(e))
        nrep = structure_replace(pred, S, C, cfg, St, tcfg, teid)
        print(f"  {c:13s} {len(teid):>5} configs  (+{nrep} structured refined)", flush=True)
    write_csv("submission.csv", pred, order)

# ----------------------------------------------------------------------------
# train: fit all 8 predictors from scratch (regime-A) -> save .pth, then reproduce
# ----------------------------------------------------------------------------
def train():
    torch, nn, F, Lift, ResBlock, AttnBlock, BigNet = _torch()
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", DEV, "| torch", torch.__version__)

    def train_seed(S, C, B, m, cf, seed):
        torch.manual_seed(seed)
        net = BigNet(B, m, cf["ch"], cf["blocks"], cf["mh"], cf["attn"]).to(DEV)
        rng = np.random.default_rng(seed); idx = rng.permutation(len(S))
        va, fi = idx[:800], idx[800:]                       # honest internal val split of TRAIN only
        Rt = torch.tensor(S, device=DEV); Ct = torch.tensor(C, device=DEV)
        ep, bs = cf["epochs"], 1024
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
        best = (1e9, None)
        for e in range(ep):
            net.train(); perm = fi[rng.permutation(len(fi))]
            for st in range(0, len(fi), bs):
                sel = perm[st:st + bs]
                loss = F.mse_loss(net(Rt[sel]), Ct[sel])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            sch.step()
            if e % 5 == 4 or e == ep - 1:
                net.eval()
                with torch.no_grad():
                    v = float(((net(Rt[va]) - Ct[va]) ** 2).mean())
                if v < best[0]:
                    best = (v, {k: val.detach().cpu().clone() for k, val in net.state_dict().items()})
        return best[1]

    models = {}
    for c in CONDS:
        S, C, cfg, St, tcfg, teid = load(c)
        cf = cfg_for(c)
        tot = cf["seeds"] * cnn_params(cf) + QUAD_PARAMS + MBLOCK_PARAMS
        assert tot <= 1_000_000, f"{c}: {tot:,} params > 1M -- shrink CFG"
        Cf = C.astype(np.float32)
        B, m = svd_basis(C); Bf, mf = B.astype(np.float32), m.astype(np.float32)
        seeds = [train_seed(S, Cf, Bf, mf, cf, sd) for sd in range(1, cf["seeds"] + 1)]
        models[c] = {"cnn_seeds": seeds, "B": torch.tensor(Bf), "mean": torch.tensor(mf)}
        print(f"  {c:13s} trained {cf['seeds']} seeds ({tot:,} params <=1M, incl. macro-block head)", flush=True)
    os.makedirs("models", exist_ok=True)
    torch.save({"models": models, "blend": {}}, MODEL_PATH)  # scalar-alpha blend by default
    print(f"saved {MODEL_PATH}")
    reproduce()

# ----------------------------------------------------------------------------
# params: print the per-predictor parameter budget
# ----------------------------------------------------------------------------
def params():
    print(f"{'predictor':9s} {'1 net':>10s} x seeds + quad + mblock = total   <= 1,000,000")
    for k in ("Log", "Dipole"):
        cf = CFG[k]; one = cnn_params(cf); tot = cf["seeds"] * one + QUAD_PARAMS + MBLOCK_PARAMS
        print(f"{k:9s} {one:>10,} x{cf['seeds']} + quad {QUAD_PARAMS:,} + mblock {MBLOCK_PARAMS:,} = {tot:>9,}   "
              f"{'OK' if tot <= 1_000_000 else 'OVER!'}")

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "reproduce"
    if   mode == "reproduce": reproduce()
    elif mode == "train":     train()
    elif mode == "params":    params()
    else:
        print(__doc__); sys.exit(0 if mode in ("-h", "--help", "help") else 1)
