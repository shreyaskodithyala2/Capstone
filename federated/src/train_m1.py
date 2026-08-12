"""Stage A: single-mission GDN on Mission1 lightweight subset (channels 41-46).
Trains a forecasting GDN on the clean-ish train split, scores the test split with
graph-deviation scoring, and evaluates with the OFFICIAL ESAScores (corrected
event-wise F0.5). Runs on Apple MPS; memory-safe (windows built on the fly)."""
import os, sys, time, argparse, contextlib, io
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/ESA-ADB")   # for timeeval metric
from gdn import GDN
from timeeval.metrics import ESAScores

PREP = "/Users/Shreyas/Downloads/Capstone/ESA/ESA-ADB/data/preprocessed/multivariate"
torch.manual_seed(42); np.random.seed(42)


class WindowDS(Dataset):
    """Forecasting windows built on the fly from an (T, N) array (no big materialization)."""
    def __init__(self, arr, w, stride=1):
        self.a = arr; self.w = w
        self.idx = np.arange(0, len(arr) - w, stride)
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        x = self.a[i:i + self.w].T            # (N, w)
        y = self.a[i + self.w]                # (N,)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_arr(csv, chs):
    df = pd.read_csv(csv, usecols=chs)
    return df[chs].to_numpy(np.float32)


def evaluate_scores(model, arr, w, device, bs=8192):
    """Return per-timestep anomaly score A(t)=max_i robust_err, plus raw |err| per sensor."""
    model.eval()
    N = arr.shape[1]
    ds = WindowDS(arr, w, stride=1)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    errs = np.zeros((len(arr), N), dtype=np.float32)   # aligned to target index i+w
    with torch.no_grad():
        pos = w
        for x, y in dl:
            x = x.to(device); y = y.to(device)
            pred = model(x)
            e = (pred - y).abs().cpu().numpy()
            errs[pos:pos + len(e)] = e
            pos += len(e)
    return errs


def smooth_errs(errs, k):
    """Causal moving-average per sensor -> connects subsequence detections, kills spikes."""
    if k <= 1:
        return errs
    ker = np.ones(k, dtype=np.float32) / k
    out = np.empty_like(errs)
    for c in range(errs.shape[1]):
        out[:, c] = np.convolve(errs[:, c], ker, mode="full")[:len(errs)]
    return out


def robust_score(errs, med, iqr, w):
    a = (errs - med) / iqr
    A = a.max(axis=1)
    A[:w] = A[w:].min()          # first w have no prediction -> nominal
    return A


def build_y_true(raw_folder, chs, split_date):
    ldf = pd.read_csv(f"{raw_folder}/labels.csv", parse_dates=["StartTime", "EndTime"])
    for c in ["StartTime", "EndTime"]:
        if getattr(ldf[c].dt, "tz", None) is not None:
            ldf[c] = ldf[c].dt.tz_localize(None)
    at = pd.read_csv(f"{raw_folder}/anomaly_types.csv")
    ldf = ldf[ldf["Channel"].isin(chs) & (ldf["EndTime"] > pd.to_datetime(split_date))]
    ldf = ldf.merge(at[["ID", "Category", "Dimensionality", "Locality", "Length"]], on="ID", how="left")
    return ldf.drop(columns=["Channel"]).drop_duplicates(subset=["ID"]).reset_index(drop=True)


def esa_f05(y_true, timestamps_dt, binary):
    yp = np.empty((len(timestamps_dt), 2), dtype=object)
    yp[:, 0] = timestamps_dt; yp[:, 1] = binary.astype(int)
    m = ESAScores(betas=0.5, select_labels={"Category": ["Rare Event", "Anomaly"]})
    with contextlib.redirect_stdout(io.StringIO()):      # silence metric's debug prints
        return m.score(y_true, yp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=100)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_windows", type=int, default=600000)   # cap => stride
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--smooth", type=int, default=10)              # error smoothing window
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(6)
    chs = [f"channel_{i}" for i in range(41, 47)]
    N = len(chs)
    coll = "ESA-Mission1-semi-supervised"
    raw = "/Users/Shreyas/Downloads/Capstone/ESA/ESA-Mission1"
    t0 = time.time()

    # ---- load + normalize (train stats) ----
    tr_full = load_arr(f"{PREP}/{coll}/84_months.train.csv", chs)
    mu, sd = tr_full.mean(0), tr_full.std(0) + 1e-8
    tr_full = (tr_full - mu) / sd
    n_val = int(0.15 * len(tr_full))
    tr, val = tr_full[:-n_val], tr_full[-n_val:]
    stride = max(1, (len(tr) - args.w) // args.train_windows)
    print(f"device={device} | train rows={len(tr)} val rows={len(val)} | stride={stride} "
          f"| ~{(len(tr)-args.w)//stride} train windows | {time.time()-t0:.0f}s", flush=True)

    model = GDN(N, args.w, embed_dim=args.embed).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"GDN params: {nparams}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.MSELoss()

    dl = DataLoader(WindowDS(tr, args.w, stride=stride), batch_size=args.bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(WindowDS(val, args.w, stride=1), batch_size=8192, shuffle=False, num_workers=0)

    best_val, best_state, bad = 1e9, None, 0
    for ep in range(args.epochs):
        model.train(); tl = 0; nb = 0
        for x, y in dl:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        # val
        model.eval(); vl = 0; vn = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device); y = y.to(device)
                vl += lossf(model(x), y).item(); vn += 1
        vloss = vl / max(vn, 1)
        print(f"  epoch {ep+1}: train_mse={tl/max(nb,1):.4f} val_mse={vloss:.4f} | {time.time()-t0:.0f}s", flush=True)
        if vloss < best_val - 1e-5:
            best_val, best_state, bad = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stop", flush=True); break
    model.load_state_dict(best_state)

    # ---- deviation scoring: robust stats from SMOOTHED validation errors ----
    val_err = smooth_errs(evaluate_scores(model, val, args.w, device), args.smooth)
    med = np.median(val_err[args.w:], axis=0)
    iqr = (np.percentile(val_err[args.w:], 75, axis=0) - np.percentile(val_err[args.w:], 25, axis=0)) + 1e-8
    A_val = robust_score(val_err, med, iqr, args.w)
    thr_pct = np.percentile(A_val, 99.9)     # robust operating point (val 99.9th pct)
    print(f"smooth={args.smooth} | val 99.9pct threshold = {thr_pct:.3f} | {time.time()-t0:.0f}s", flush=True)

    # ---- test ----
    te = (load_arr(f"{PREP}/{coll}/84_months.test.csv", chs) - mu) / sd
    ts = pd.to_datetime(pd.read_csv(f"{PREP}/{coll}/84_months.test.csv", usecols=["timestamp"])["timestamp"])
    A_te = robust_score(smooth_errs(evaluate_scores(model, te, args.w, device), args.smooth), med, iqr, args.w)
    print(f"test scored; A range [{A_te.min():.2f},{A_te.max():.2f}] | {time.time()-t0:.0f}s", flush=True)

    y_true = build_y_true(raw, chs, "2007-01-01")
    print(f"y_true events: {len(y_true)} (Anomaly={(y_true.Category=='Anomaly').sum()}, "
          f"RareEvent={(y_true.Category=='Rare Event').sum()})", flush=True)

    # proper (no test-peeking) threshold: val 99.9th percentile
    res_valthr = esa_f05(y_true, ts, (A_te > thr_pct))
    print(f"\n[VAL-THRESHOLD] EW_F0.5={res_valthr['EW_F_0.50']:.4f} "
          f"(P={res_valthr['EW_precision']:.3f} R={res_valthr['EW_recall']:.3f}) pos={int((A_te>thr_pct).sum())}", flush=True)

    # oracle ceiling: best F0.5 over a threshold sweep (labeled, for diagnosis only)
    best = {"EW_F_0.50": -1}; best_thr = 0
    qs = np.quantile(A_te, [0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 0.99995])
    for thr in qs:
        r = esa_f05(y_true, ts, (A_te > thr))
        if r["EW_F_0.50"] > best["EW_F_0.50"]:
            best, best_thr = r, thr
    print(f"[ORACLE-BEST]   EW_F0.5={best['EW_F_0.50']:.4f} "
          f"(P={best['EW_precision']:.3f} R={best['EW_recall']:.3f}) @thr={best_thr:.3f}", flush=True)
    print(f"\nDONE total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
