"""Train GDNv2 on one mission; tune ALL detection knobs on the TRAIN period
(labels there are legitimately available in the semi-supervised setting);
pre-register the best train-period operating point; report it on TEST.
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_arrays, timestamps, gap_all_mask, MISSIONS
from model import GDNv2, RF, series_errors
from score import ewma, gap_pred
sys.path.insert(0, "/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/src")
from official_score import official_score
from telemanom_prune import telemanom_binary_multichannel

torch.manual_seed(0); np.random.seed(0)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = "/Users/Shreyas/Downloads/Capstone/ESA/GDNv2/results"


def sample_batch(X, starts, seg, n_ch, rng, bs):
    ss = rng.choice(starts, bs, replace=False)
    xs = np.stack([np.transpose(X[s:s + seg], (1, 0, 2)) for s in ss])   # (B, N, L, F)
    ys = np.stack([X[s + RF:s + seg, :n_ch, 0] for s in ss])             # (B, L-RF, n_ch)
    return torch.from_numpy(xs), torch.from_numpy(ys)


def train_model(X, n_ch, n_tc, args, vsplit):
    model = GDNv2(n_ch, n_tc, topk=args.topk).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.MSELoss()
    rng = np.random.default_rng(0)
    tr_starts = np.arange(0, vsplit - args.seg - 1)
    va_starts = np.arange(vsplit, len(X) - args.seg - 1, args.seg)[:80]
    best, bad, best_state = 1e9, 0, None
    for ep in range(args.epochs):
        model.train(); tl = 0.0
        for _ in range(args.batches):
            x, y = sample_batch(X, tr_starts, args.seg, n_ch, rng, args.bs)
            opt.zero_grad()
            loss = lossf(model.forward_segments(x.to(DEV)), y.to(DEV))
            loss.backward(); opt.step(); tl += loss.item()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for i in range(0, len(va_starts), args.bs):
                ss = va_starts[i:i + args.bs]
                xs = np.stack([np.transpose(X[s:s + args.seg], (1, 0, 2)) for s in ss])
                ys = np.stack([X[s + RF:s + args.seg, :n_ch, 0] for s in ss])
                vl += lossf(model.forward_segments(torch.from_numpy(xs).to(DEV)),
                            torch.from_numpy(ys).to(DEV)).item() * len(ss)
        vl /= len(va_starts)
        print(f"  ep{ep+1}: train={tl/args.batches:.5f} val={vl:.5f}", flush=True)
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stop", flush=True); break
    model.load_state_dict(best_state)
    return model


def operating_points(z, errs_s, vals, gap_bin, thresholds, p, n_ignore):
    out = {}
    for name, thr in thresholds.items():
        b = (z > thr).astype(np.uint8)
        if b.sum():
            out[name] = b
    pr = telemanom_binary_multichannel(errs_s, vals, p=p, num_to_ignore=n_ignore)
    if pr.sum():
        out["prune"] = pr
    if gap_bin is not None and gap_bin.sum():
        for k in list(out):
            out[k + "+gap"] = (out[k] | gap_bin).astype(np.uint8)
        out["gap_only"] = gap_bin
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="M1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--span", type=int, default=30)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--gapL", type=int, default=10)
    ap.add_argument("--p", type=float, default=0.13)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    mk = args.mission; m = MISSIONS[mk]; t0 = time.time()

    Xtr, mu, sd, n_ch, n_tc = build_arrays(mk, "train")
    Xte, _, _, _, _ = build_arrays(mk, "test", mu, sd)
    vsplit = int(0.85 * len(Xtr))
    print(f"{mk}: T_train={len(Xtr):,} T_test={len(Xte):,} n_ch={n_ch} n_tc={n_tc} dev={DEV}", flush=True)

    model = train_model(Xtr, n_ch, n_tc, args, vsplit)
    torch.save(model.state_dict(), f"{OUT}/{mk}_{args.tag}.pt")
    print(f"trained in {time.time()-t0:.0f}s", flush=True)

    err_tr = series_errors(model, Xtr, DEV)
    err_te = series_errors(model, Xte, DEV)
    print(f"error passes done {time.time()-t0:.0f}s", flush=True)

    es_tr, es_te = ewma(err_tr, args.span), ewma(err_te, args.span)
    med = np.median(es_tr[vsplit:], 0)
    iqr = np.percentile(es_tr[vsplit:], 75, 0) - np.percentile(es_tr[vsplit:], 25, 0) + 1e-8
    z_tr = ((es_tr - med) / iqr).max(1); z_tr[:RF] = 0
    z_te = ((es_te - med) / iqr).max(1); z_te[:RF] = 0

    qs = [0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 0.99995]
    thr = {f"q{q}": float(np.quantile(z_tr, q)) for q in qs}

    gap_tr = gap_pred(gap_all_mask(mk, "train"), args.gapL) if mk == "M1" else None
    gap_te = gap_pred(gap_all_mask(mk, "test"), args.gapL) if mk == "M1" else None

    ts_tr, ts_te = timestamps(mk, "train"), timestamps(mk, "test")
    raw = m["raw"]; chs = m["chs"]

    ops_tr = operating_points(z_tr, es_tr, Xtr[:, :n_ch, 0], gap_tr, thr, args.p, RF)
    print("\n=== TRAIN-period official F0.5 (tuning signal) ===", flush=True)
    tr_scores = {}
    for name, b in ops_tr.items():
        r = official_score(b, ts_tr, f"{raw}/labels.csv", f"{raw}/anomaly_types.csv", chs)
        tr_scores[name] = r["EW_F_0.50"]
        print(f"  {name:>14}: F0.5={r['EW_F_0.50']:.4f} (P={r['EW_precision']:.3f} R={r['EW_recall']:.3f})", flush=True)
    best_mode = max(tr_scores, key=tr_scores.get)
    print(f"\n>> PRE-REGISTERED mode (best on train): {best_mode}", flush=True)

    ops_te = operating_points(z_te, es_te, Xte[:, :n_ch, 0], gap_te, thr, args.p, RF)
    print("\n=== TEST official F0.5 ===", flush=True)
    results = {}
    for name, b in ops_te.items():
        r = official_score(b, ts_te, f"{raw}/labels.csv", f"{raw}/anomaly_types.csv", chs)
        results[name] = {k: float(v) for k, v in r.items()}
        star = "  <-- HEADLINE" if name == best_mode else ""
        print(f"  {name:>14}: F0.5={r['EW_F_0.50']:.4f} (P={r['EW_precision']:.3f} R={r['EW_recall']:.3f}){star}", flush=True)

    json.dump({"mission": mk, "tag": args.tag, "args": vars(args), "train_scores": tr_scores,
               "best_mode": best_mode, "test": results},
              open(f"{OUT}/{mk}_{args.tag}.json", "w"), indent=1)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
