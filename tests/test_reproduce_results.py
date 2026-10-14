"""Re-computes the headline results from the saved models and checks them against the saved
result files, so any code change that alters a published number is caught.

  Mission 1 (results/M1_result.json, F0.5 0.8532): forecast errors from results/M1_model.pt,
      the saved frozen detection config applied to train and test, every ESA metric compared.
  Mission 2 (results/M2_result.json, F0.5 0.9644): the full absolute-z rule (train sweep of
      z*, then one test evaluation), compared with the saved z*, train and test scores.

It skips the 432-config Mission 1 grid search (the saved config is the one it selected) but
checks that config's train F0.5. About 11 minutes on an M-series Mac.

    python tests/test_reproduce_results.py
"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from config import MISSIONS, DATA, RESULTS
from dataset import build_arrays, timestamps
from model import GDNForecaster, RF, series_errors
from detectors import multi_span_z, ewma, gap_pred_ch, zero_gap_margin
from esa_metrics import full_scoreboard, global_f05
from run_m1_lightweight import assemble, prune_per_channel, SPANS
import run_m2_lightweight as m2

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
TOL = 1e-12


def same(name, got, exp):
    """Assert two numbers are equal (to TOL) and print the check."""
    assert abs(float(got) - float(exp)) <= TOL, (name, got, exp)
    print(f"  OK  {name:34s} {float(got):.6f}")


def check_m1():
    print("Mission 1 lightweight")
    saved = json.load(open(f"{RESULTS}/M1_result.json"))
    cfg, block = saved["config"], saved["block"]
    m = MISSIONS["M1"]; RAW, CHS = m["raw"], m["chs"]

    Xtr, mu, sd, n_ch, n_tc = build_arrays("M1", "train")
    Xte, *_ = build_arrays("M1", "test", mu, sd)
    model = GDNForecaster(n_ch, n_tc).to(DEV)
    model.load_state_dict(torch.load(f"{RESULTS}/M1_model.pt", map_location=DEV))

    P = {}
    for split, X in (("train", Xtr), ("test", Xte)):
        f = np.load(f"{DATA}/M1_{split}_feats.npz")
        ec = zero_gap_margin(series_errors(model, X, DEV), (f["count"] == 0).all(1), 30)
        Z = multi_span_z(ec, SPANS, block); Z[:RF] = 0               # saved config: local z
        if split == "train":
            thr = float(np.quantile(Z, cfg["q"]))                    # fitted on TRAIN only
            same("threshold (train quantile)", thr, saved["thr_value"])
        A = prune_per_channel(ewma(ec, 5), X[:, :n_ch, 0], cfg["p"], RF)
        C = gap_pred_ch(f["count"], cfg["gapL"])
        P[split] = assemble(A, Z, thr, C, f["tcdt"].min(1) <= 7200.0, cfg["boost"],
                            cfg["close"], cfg["minlen"])

    tr = global_f05(P["train"], timestamps("M1", "train"), RAW, CHS)
    same("train event-wise F0.5", tr["global_EW_F_0.50"], saved["train_f05"])
    sb = full_scoreboard(P["test"], timestamps("M1", "test"), RAW, CHS)
    for k, v in saved["test"].items():                              # every saved metric
        if isinstance(v, (int, float)):
            same(f"test {k}", sb[k], v)


def check_m2():
    print("Mission 2 lightweight")
    saved = json.load(open(f"{RESULTS}/M2_result.json"))
    Xtr, mu, sd, n_ch, n_tc = build_arrays("M2", "train")
    Xte, *_ = build_arrays("M2", "test", mu, sd)
    model = GDNForecaster(n_ch, n_tc).to(DEV)
    model.load_state_dict(torch.load(f"{RESULTS}/M2_model.pt", map_location=DEV))
    z_tr = m2.zseries(series_errors(model, Xtr, DEV), m2.SPAN)
    z_te = m2.zseries(series_errors(model, Xte, DEV), m2.SPAN)
    ts_tr, ts_te = timestamps("M2", "train"), timestamps("M2", "test")

    best = (None, -1)                        # same train sweep as run_m2_lightweight.main
    for zstar in m2.ZSTARS:
        b = (z_tr > zstar).astype(np.uint8)
        if b.sum():
            f = m2.score(b, ts_tr)["EW_F_0.50"]
            if f > best[1]:
                best = (zstar, f)
    assert best[0] == saved["zstar"], (best[0], saved["zstar"])
    print(f"  OK  {'chosen z*':34s} {best[0]}")
    same("train event-wise F0.5", best[1], saved["train_f05"])
    r = m2.score((z_te > best[0]).astype(np.uint8), ts_te)
    for k, v in saved["test"].items():
        same(f"test {k}", r[k], v)


if __name__ == "__main__":
    t0 = time.time()
    check_m1()
    check_m2()
    print(f"\nALL RESULTS REPRODUCED ({time.time() - t0:.0f}s)")
