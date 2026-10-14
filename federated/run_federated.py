"""Federated-learning prototype across two real spacecraft: Mission 1 (6 channels) and
Mission 2 (11 channels) are the two clients.

What it compares (same total training epochs for both):
  SOLO       each mission trains its own model alone
  FEDERATED  rounds of: each client trains locally (FedProx) -> sends ONLY its shared weights
             -> server averages them (trimmed mean) -> sends the average back.
             Then each client fine-tunes on its own data (personalisation).
Embedding tables never leave a client, so 6 vs 11 channels is not a problem, and no raw
telemetry is ever shared.

Measured (val-tuned threshold, ESA event-wise F0.5, logs/federated_prototype.log):
    Mission 1: solo 0.0799 -> federated 0.0176     Mission 2: solo 0.9233 -> federated 0.9362
This prototype uses the earlier window model (window_gdn.py); porting the same loop to the
current model (src/model.py, same shared_state() interface) is the work in progress.

    python federated/run_federated.py          # ~16 min on Apple MPS
"""
import os, sys, time, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from config import MISSIONS, PREP                                  # noqa: E402
from esa_metrics import global_f05                                 # noqa: E402
from window_gdn import WindowGDN                                   # noqa: E402

torch.manual_seed(0); np.random.seed(0)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------- data ----
class WindowDS(Dataset):
    """Sliding windows: x = arr[i : i+w].T (N, w), y = arr[i+w] (N,) -- the next step."""
    def __init__(self, arr, w, stride=1):
        self.a, self.w = arr, w
        self.idx = np.arange(0, len(arr) - w, stride)
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        return torch.from_numpy(self.a[i:i + self.w].T), torch.from_numpy(self.a[i + self.w])


def load_arr(csv, chs):
    """Channel values from an official preprocessed CSV -> (T, N) float32."""
    return pd.read_csv(csv, usecols=chs)[chs].to_numpy(np.float32)


def load_mission(key, val_frac=0.15):
    """One client's data: train split z-scored with its own stats, last 15 % = validation."""
    m = MISSIONS[key]
    arr = load_arr(f"{PREP}/{m['coll']}/{m['train']}", m["chs"])
    mu, sd = arr.mean(0), arr.std(0) + 1e-8
    arr = (arr - mu) / sd
    n_val = int(val_frac * len(arr))
    return dict(key=key, cfg=m, mu=mu, sd=sd, tr=arr[:-n_val], val=arr[-n_val:], N=len(m["chs"]))


def make_loader(arr, w, train_windows, bs):
    """Shuffled loader over ~train_windows evenly spaced training windows."""
    stride = max(1, (len(arr) - w) // train_windows)
    return DataLoader(WindowDS(arr, w, stride=stride), batch_size=bs, shuffle=True)


# ------------------------------------------------------------- scoring ----
def evaluate_errors(model, arr, w, device, bs=8192):
    """|prediction - actual| at every step of arr -> (T, N); first w rows stay 0."""
    model.eval(); N = arr.shape[1]
    dl = DataLoader(WindowDS(arr, w, stride=1), batch_size=bs, shuffle=False)
    errs = np.zeros((len(arr), N), dtype=np.float32); pos = w
    with torch.no_grad():
        for x, y in dl:
            e = (model(x.to(device)) - y.to(device)).abs().cpu().numpy()
            errs[pos:pos + len(e)] = e; pos += len(e)
    return errs


def smooth_errs(errs, k):
    """Trailing moving average of length k, per channel."""
    if k <= 1: return errs
    ker = np.ones(k, np.float32) / k
    out = np.empty_like(errs)
    for c in range(errs.shape[1]):
        out[:, c] = np.convolve(errs[:, c], ker, mode="full")[:len(errs)]
    return out


def robust_score(errs, med, iqr, w):
    """One anomaly score per step: max over channels of (error - median) / IQR."""
    A = ((errs - med) / iqr).max(axis=1)
    A[:w] = A[w:].min()
    return A


def event_f05(binary, ts, cfg):
    """ESA event-wise scores of a 1-D 0/1 prediction (keys EW_F_0.50, EW_precision, ...)."""
    r = global_f05(np.asarray(binary)[:, None], ts, cfg["raw"], cfg["chs"])
    return {k.replace("global_", "", 1): v for k, v in r.items()}


def score_on_test(model, mstate, w, device, smooth=10):
    """Test F0.5 with a threshold chosen on the VALIDATION split (no test labels used).

    Steps: validation errors -> median/IQR -> scores; try 7 quantile thresholds on validation;
    apply the best one to the test scores; score with ESA's metric.
    """
    cfg, mu, sd = mstate["cfg"], mstate["mu"], mstate["sd"]
    val_err = smooth_errs(evaluate_errors(model, mstate["val"], w, device), smooth)
    med = np.median(val_err[w:], axis=0)
    iqr = (np.percentile(val_err[w:], 75, axis=0) - np.percentile(val_err[w:], 25, axis=0)) + 1e-8
    A_val = robust_score(val_err, med, iqr, w)
    tr_ts = pd.to_datetime(pd.read_csv(f"{PREP}/{cfg['coll']}/{cfg['train']}", usecols=["timestamp"])["timestamp"])
    val_ts = tr_ts.iloc[-len(mstate["val"]):].reset_index(drop=True)   # last rows of train

    te = (load_arr(f"{PREP}/{cfg['coll']}/{cfg['test']}", cfg["chs"]) - mu) / sd   # TRAIN stats
    ts = pd.to_datetime(pd.read_csv(f"{PREP}/{cfg['coll']}/{cfg['test']}", usecols=["timestamp"])["timestamp"])
    A_te = robust_score(smooth_errs(evaluate_errors(model, te, w, device), smooth), med, iqr, w)

    best_thr, best_valf = None, -1
    for q in [0.980, 0.990, 0.995, 0.998, 0.999, 0.9995, 0.9999]:
        thr = np.quantile(A_val, q)
        f = event_f05((A_val > thr).astype(int), val_ts, cfg)["EW_F_0.50"]
        if f > best_valf:
            best_valf, best_thr = f, thr
    test = event_f05((A_te > best_thr).astype(int), ts, cfg)
    test["_val_f05"] = best_valf
    return test


# ---------------------------------------------------- federated learning ----
def local_train(model, dl, epochs, lr, mu_prox=0.0, global_shared=None):
    """Client update. With mu_prox > 0 this is FedProx: the loss gets
    0.5 * mu * ||shared weights - global weights||^2, which stops a client drifting far from
    the global model on its own (non-IID) data."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    gs = {k: v.to(DEVICE) for k, v in global_shared.items()} if (global_shared and mu_prox > 0) else None
    model.train()
    for _ in range(epochs):
        for x, y in dl:
            opt.zero_grad()
            loss = lossf(model(x.to(DEVICE)), y.to(DEVICE))          # next-step forecast error
            if gs is not None:
                params = dict(model.named_parameters())
                prox = sum(((params[k] - v) ** 2).sum() for k, v in gs.items() if k in params)
                loss = loss + 0.5 * mu_prox * prox                   # FedProx term
            loss.backward(); opt.step()


def trimmed_mean(states, trim=0):
    """Server aggregation: per weight, drop the `trim` highest and lowest client values, then
    average (trim=0 is the plain mean, FedAvg-style). Input: list of shared-weight dicts."""
    out = {}
    for k in states[0]:
        stack = torch.stack([s[k].float() for s in states], dim=0)
        if trim > 0 and stack.size(0) > 2 * trim:
            stack, _ = torch.sort(stack, dim=0)
            stack = stack[trim:stack.size(0) - trim]
        out[k] = stack.mean(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=100, help="input window length")
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=6, help="federated rounds")
    ap.add_argument("--local_epochs", type=int, default=2, help="client epochs per round")
    ap.add_argument("--ft_epochs", type=int, default=4, help="personalisation fine-tune epochs")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mu", type=float, default=0.01, help="FedProx strength")
    ap.add_argument("--trim", type=int, default=0, help="trimmed mean (0 = mean; needs > 2 clients)")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--train_windows", type=int, default=300000)
    ap.add_argument("--smooth", type=int, default=10)
    args = ap.parse_args()
    E_total = args.rounds * args.local_epochs + args.ft_epochs   # SOLO gets the same budget
    t0 = time.time()
    print(f"device={DEVICE} | fed: {args.rounds} rounds x {args.local_epochs} local + {args.ft_epochs} finetune "
          f"= {E_total} epochs (SOLO gets {E_total}) | FedProx mu={args.mu} trim={args.trim}", flush=True)

    M = {k: load_mission(k) for k in ["M1", "M2"]}                 # the two clients
    for k in M:
        M[k]["dl"] = make_loader(M[k]["tr"], args.w, args.train_windows, args.bs)
        print(f"  {k}: N={M[k]['N']} channels, train rows={len(M[k]['tr'])}, "
              f"~{len(M[k]['dl'].dataset)} windows", flush=True)

    # ---- SOLO baselines ----
    print("\n=== SOLO (each mission trained independently) ===", flush=True)
    solo = {}
    for k in ["M1", "M2"]:
        torch.manual_seed(0)
        model = WindowGDN(M[k]["N"], args.w, embed_dim=args.embed).to(DEVICE)
        local_train(model, M[k]["dl"], E_total, args.lr)
        res = score_on_test(model, M[k], args.w, DEVICE, smooth=args.smooth)
        solo[k] = res["EW_F_0.50"]
        print(f"  SOLO {k}: EW_F0.5={res['EW_F_0.50']:.4f} (P={res['EW_precision']:.3f} "
              f"R={res['EW_recall']:.3f}) | {time.time()-t0:.0f}s", flush=True)

    # ---- FEDERATED ----
    print("\n=== FEDERATED (cross-mission: shared operators, personalized embeddings) ===", flush=True)
    torch.manual_seed(0)
    models = {k: WindowGDN(M[k]["N"], args.w, embed_dim=args.embed).to(DEVICE) for k in ["M1", "M2"]}
    global_shared = {k: v.cpu().clone() for k, v in models["M1"].shared_state().items()}
    for k in models:
        models[k].load_shared_state(global_shared)                  # common starting point

    for r in range(args.rounds):
        updates = []
        for k in ["M1", "M2"]:                                       # each client, in turn
            models[k].load_shared_state(global_shared)               # download global weights
            local_train(models[k], M[k]["dl"], args.local_epochs, args.lr,
                        mu_prox=args.mu, global_shared=global_shared)
            updates.append({kk: v.detach().cpu().clone() for kk, v in models[k].shared_state().items()})
        global_shared = trimmed_mean(updates, trim=args.trim)        # server aggregation
        print(f"  round {r+1}/{args.rounds} aggregated | {time.time()-t0:.0f}s", flush=True)

    # personalisation: each client fine-tunes the global weights on its own data
    fed = {}
    for k in ["M1", "M2"]:
        models[k].load_shared_state(global_shared)
        if args.ft_epochs > 0:
            local_train(models[k], M[k]["dl"], args.ft_epochs, args.lr)
        res = score_on_test(models[k], M[k], args.w, DEVICE, smooth=args.smooth)
        fed[k] = res["EW_F_0.50"]
        print(f"  FED  {k}: EW_F0.5={res['EW_F_0.50']:.4f} (P={res['EW_precision']:.3f} "
              f"R={res['EW_recall']:.3f}) | {time.time()-t0:.0f}s", flush=True)

    print("\n================ RESULT (EW corrected event-wise F0.5) ================", flush=True)
    print(f"{'mission':<8}{'SOLO':>10}{'FEDERATED':>12}{'delta':>10}", flush=True)
    for k in ["M1", "M2"]:
        print(f"{k:<8}{solo[k]:>10.4f}{fed[k]:>12.4f}{fed[k] - solo[k]:>+10.4f}", flush=True)
    print(f"\nDONE total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
