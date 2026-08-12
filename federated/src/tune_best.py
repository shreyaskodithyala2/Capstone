"""Best comparable GDN numbers: M1 via Telemanom-pruning with p tuned on VALIDATION
(no test labels), M2 via val-threshold. Trains each mission once; errors computed once."""
import os, sys, time, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from gdn import GDN
from common import (load_mission, WindowDS, PREP, evaluate_scores, smooth_errs,
                    robust_score, official_score, score_on_test_official)
from telemanom_prune import telemanom_binary_multichannel

torch.manual_seed(0); np.random.seed(0)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def train(M, w, embed, epochs, lr, bs, train_windows):
    stride = max(1, (len(M["tr"]) - w) // train_windows)
    dl = DataLoader(WindowDS(M["tr"], w, stride=stride), batch_size=bs, shuffle=True)
    torch.manual_seed(0)
    model = GDN(M["N"], w, embed_dim=embed).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = torch.nn.MSELoss(); model.train()
    for _ in range(epochs):
        for x, y in dl:
            opt.zero_grad(); lossf(model(x.to(DEVICE)), y.to(DEVICE)).backward(); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=100); ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=16); ap.add_argument("--smooth", type=int, default=10)
    args = ap.parse_args(); t0 = time.time()

    for k in ["M1", "M2"]:
        M = load_mission(k); cfg = M["cfg"]; chs = cfg["chs"]
        labels_csv = f"{cfg['raw']}/labels.csv"; at_csv = f"{cfg['raw']}/anomaly_types.csv"
        model = train(M, args.w, args.embed, args.epochs, 1e-3, 256, 300000)

        # errors + values + timestamps for val and test (computed once)
        te = (np.asarray(pd.read_csv(f"{PREP}/{cfg['coll']}/{cfg['test']}", usecols=chs)[chs], np.float32) - M["mu"]) / M["sd"]
        ts = pd.to_datetime(pd.read_csv(f"{PREP}/{cfg['coll']}/{cfg['test']}", usecols=["timestamp"])["timestamp"])
        te_err = smooth_errs(evaluate_scores(model, te, args.w, DEVICE), args.smooth)
        val = M["val"]; val_err = smooth_errs(evaluate_scores(model, val, args.w, DEVICE), args.smooth)
        n_val = len(val)
        val_ts = pd.to_datetime(pd.read_csv(f"{PREP}/{cfg['coll']}/{cfg['train']}", usecols=["timestamp"])["timestamp"]).iloc[-n_val:].reset_index(drop=True)

        # sweep p on validation Telemanom-pruning
        best_p, best_vf = None, -1
        for p in [0.01, 0.02, 0.04, 0.07, 0.10, 0.13, 0.20, 0.30]:
            vb = telemanom_binary_multichannel(val_err, val, p=p, num_to_ignore=args.w)
            if vb.sum() == 0:
                continue
            vf = official_score(vb, val_ts, labels_csv, at_csv, chs)["EW_F_0.50"]
            if vf > best_vf:
                best_vf, best_p = vf, p
        # test at val-selected p
        if best_p is not None:
            tb = telemanom_binary_multichannel(te_err, te, p=best_p, num_to_ignore=args.w)
            rt = official_score(tb, ts, labels_csv, at_csv, chs)
            prune_str = f"PRUNE(p={best_p},valF={best_vf:.3f}) EW_F0.5={rt['EW_F_0.50']:.4f} (P={rt['EW_precision']:.3f} R={rt['EW_recall']:.3f})"
        else:
            prune_str = "PRUNE: no positives at any p"
        # val-threshold method
        rv = score_on_test_official(model, M, args.w, DEVICE, smooth=args.smooth)
        print(f"{k}: {prune_str}  |  VAL-THRESH EW_F0.5={rv['EW_F_0.50']:.4f} "
              f"(P={rv['EW_precision']:.3f} R={rv['EW_recall']:.3f}) | {time.time()-t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
