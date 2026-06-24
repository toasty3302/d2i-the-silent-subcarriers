"""Preprocess the official data into the per-condition .npy arrays that
the_silent_subcarriers_0.py reads. One condition = (antenna_type, position_id):
Dipole/Log x positions {1,2,3,5} -> 8 conditions.

TWO EQUIVALENT DATA SOURCES (same underlying BRISC measurements) via --source:
  kaggle : the official Task-0 Kaggle CSVs (train.csv + test.csv).          [default]
  brisc  : the official BRISC .mat dataset (configurations_10000.mat +
           antenna{ant}_pos{pos}.mat) using the official loader convention.
The Kaggle CSVs are just the organizers' CSV repackaging of the BRISC .mat: the
per-config RIS bits are bit-identical and the per-config CSI is the frame-average of
the .mat (verified: RIS exact, CSI relative error ~2e-8). Either source therefore yields
the SAME .npy arrays -> same model -> same submission. config_id is 1-based; the RIS /
CSI for config_id k live at row k-1 of configurations_10000.mat / the averaged .mat.

Usage:
  python prep_data.py                                   # kaggle (defaults: kaggle_data -> data)
  python prep_data.py <kaggle_dir> <out_dir>            # kaggle, explicit dirs (back-compat)
  python prep_data.py --source brisc                    # BRISC .mat -> data
  python prep_data.py --source brisc --brisc-root ../ISIT2026-challenge-dataset --out data

Writes, per condition c:
  train_{c}_ris.npy (N x 256 int8)  train_{c}_csi.npy (N x 484 f32)  train_{c}_cfg.npy (N int)
  test_{c}_ris.npy  (M x 256 int8)  test_{c}_eid.npy  (M object)     test_{c}_cfg.npy  (M int)
Only training-config CSI is ever read; never held-out test channels, never private positions.
"""
import numpy as np, csv, sys, os, argparse

ANTS, POSITIONS = ["Dipole", "Log"], [1, 2, 3, 5]
N_CFG, N_SUB = 10000, 242                      # configs per condition, OFDM subcarriers


# --------------------------------------------------------------- source: Kaggle CSVs
def prep_kaggle(kaggle_dir, out):
    """Original path: read the per-condition RIS+CSI straight from the Kaggle CSVs."""
    def process(fn, has_csi):
        with open(os.path.join(kaggle_dir, fn)) as f:
            rd = csv.reader(f); hdr = next(rd)
            ris_i = [i for i, h in enumerate(hdr) if h.startswith("ris_")]
            csi_i = [i for i, h in enumerate(hdr) if h.startswith("csi_")]  # real_000..241 then imag_000..241
            eid_i = hdr.index("example_id"); at_i = hdr.index("antenna_type")
            pos_i = hdr.index("position_id"); cfg_i = hdr.index("config_id")
            rows = {}
            for r in rd:
                rows.setdefault(f"{r[at_i]}_pos{r[pos_i]}", []).append(r)
        kind = "train" if has_csi else "test"
        for c, rs in sorted(rows.items()):
            np.save(f"{out}/{kind}_{c}_ris.npy",
                    np.array([[int(r[i]) for i in ris_i] for r in rs], dtype=np.int8))
            np.save(f"{out}/{kind}_{c}_cfg.npy",
                    np.array([int(r[cfg_i]) for r in rs]))
            if has_csi:
                np.save(f"{out}/train_{c}_csi.npy",
                        np.array([[float(r[i]) for i in csi_i] for r in rs], dtype=np.float32))
            else:
                np.save(f"{out}/test_{c}_eid.npy",
                        np.array([r[eid_i] for r in rs], dtype=object))
            print(f"  {c:13} {len(rs):>5} rows ({kind})")
    print("train.csv ->"); process("train.csv", True)
    print("test.csv  ->"); process("test.csv", False)


# --------------------------------------------------------------- source: BRISC .mat
def _config_codebook(brisc_root):
    """(10000, 256) int8 RIS bits, F-order flatten matching the 1-based confId index."""
    from scipy.io import loadmat
    m = np.asarray(loadmat(os.path.join(brisc_root, "configurations_10000.mat"))["matrices"])
    if m.shape != (N_CFG, 16, 16):
        raise ValueError(f"expected ({N_CFG},16,16) configs, got {m.shape}")
    return m.transpose(0, 2, 1).reshape(N_CFG, -1).astype(np.int8)


def _avg_csi(mat_path, chunk=32768):
    """Per-config frame-AVERAGED complex CSI over all subcarriers: (10000, 242) + counts."""
    import h5py
    sums = np.zeros((N_CFG, N_SUB), np.complex128); cnt = np.zeros(N_CFG, np.int64)
    with h5py.File(mat_path, "r") as fh:
        csi, conf = fh["csi"], fh["confId"]
        for s in range(0, csi.shape[0], chunk):
            e = min(s + chunk, csi.shape[0])
            c = np.asarray(conf[s:e]).reshape(-1).astype(np.int64) - 1      # confId is 1-based
            blk = csi[s:e]; v = (blk["real"] + 1j * blk["imag"]).reshape(e - s, N_SUB)
            keep = (c >= 0) & (c < N_CFG)
            np.add.at(sums, c[keep], v[keep]); np.add.at(cnt, c[keep], 1)
    avg = np.zeros((N_CFG, N_SUB), np.complex128); pr = cnt > 0
    avg[pr] = sums[pr] / cnt[pr][:, None]
    return avg, cnt


def _test_list(kaggle_dir):
    """Which configs to predict + their submission example_ids, per condition.
    Read from the small RIS-only test.csv (no CSI answers -> leakage-free); at official
    grading the loader supplies the same target list."""
    tl = {}
    with open(os.path.join(kaggle_dir, "test.csv")) as f:
        rd = csv.reader(f); hdr = next(rd)
        eid_i = hdr.index("example_id"); at_i = hdr.index("antenna_type")
        pos_i = hdr.index("position_id"); cfg_i = hdr.index("config_id")
        for r in rd:
            tl.setdefault(f"{r[at_i]}_pos{r[pos_i]}", []).append((int(r[cfg_i]), r[eid_i]))
    return tl


def prep_brisc(brisc_root, out, kaggle_dir, conditions=None):
    """BRISC path: same arrays as Kaggle, built from the .mat files. CSI = frame-average of
    the .mat; train set = all measured configs MINUS the test configs (= the Kaggle split)."""
    S = _config_codebook(brisc_root)
    test_list = _test_list(kaggle_dir)                         # target configs + example_ids
    conds = conditions or [f"{a}_pos{p}" for a in ANTS for p in POSITIONS]
    for c in conds:
        ant, pos = c.split("_pos")
        avg, cnt = _avg_csi(os.path.join(brisc_root, f"antenna{ant}_pos{pos}.mat"))
        present = set(np.where(cnt > 0)[0].tolist())           # 0-based present configs
        tcfg = np.array([cid for cid, _ in test_list.get(c, [])], dtype=int)   # 1-based
        teid = np.array([eid for _, eid in test_list.get(c, [])], dtype=object)
        test0 = tcfg - 1                                       # 0-based test indices
        train0 = np.array(sorted(present - set(test0.tolist())), dtype=int)
        csi484 = lambda idx: np.concatenate([avg[idx].real, avg[idx].imag], axis=1).astype(np.float32)
        np.save(f"{out}/train_{c}_ris.npy", S[train0])
        np.save(f"{out}/train_{c}_csi.npy", csi484(train0))
        np.save(f"{out}/train_{c}_cfg.npy", (train0 + 1).astype(int))         # 1-based config_id
        np.save(f"{out}/test_{c}_ris.npy", S[test0])
        np.save(f"{out}/test_{c}_cfg.npy", tcfg)
        np.save(f"{out}/test_{c}_eid.npy", teid)
        print(f"  {c:13} train {len(train0):>5} / test {len(test0):>4} (brisc .mat)")


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kaggle_dir", nargs="?", default="kaggle_data")
    ap.add_argument("out", nargs="?", default="data")
    ap.add_argument("--source", choices=["kaggle", "brisc"], default="kaggle")
    ap.add_argument("--brisc-root", default="../ISIT2026-challenge-dataset",
                    help="dir with configurations_10000.mat + antenna{ant}_pos{pos}.mat")
    ap.add_argument("--conditions", nargs="+", default=None, help="brisc: subset e.g. Dipole_pos1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.source == "kaggle":
        print(f"[prep] source=kaggle  dir={a.kaggle_dir} -> {a.out}/")
        prep_kaggle(a.kaggle_dir, a.out)
    else:
        print(f"[prep] source=brisc   root={a.brisc_root}  (target list from {a.kaggle_dir}/test.csv) -> {a.out}/")
        prep_brisc(a.brisc_root, a.out, a.kaggle_dir, a.conditions)
    print(f"done -> {a.out}/")
