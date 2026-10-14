"""Full channel sets (M1 = 76 channels, M2 = 100): same detection stack as the lightweight
runner, scaled up. This is ESA-ADB's default setting; it runs on Kaggle (see kaggle/).

Mission 1 full result (results/M1_full_result.json): event-wise F0.5 0.4394 vs best published
0.061; alarming precision 0.8085 vs 0.206. Config chosen on train: p=0.3, zmode=global,
tmode=perchan, q=0.99999, close=60, minlen=3, gapL=10, gapmult=6, boost=0.6.

Differences from run_m1_lightweight.py (all needed only at this scale):
  1. Phased: tune on TRAIN, save the chosen config, free memory, then build TEST
     (~13 GB peak instead of ~45 GB).
  2. Only Target=YES channels are scored (58/76 on M1, 47/100 on M2). ESA labels no other
     channel, so an alarm there could only count as a false positive. All channels are
     still model inputs.
  3. Per-channel thresholds are an option (tmode="perchan"): with 76 channels one global
     quantile is owned by the noisiest channels and quiet ones can never cross it.
  4. Correct whole-series z (detectors.global_multi_span_z) for zmode global/max.
  5. "No pruning" (p=None) is in the grid; train labels decide whether Detector A helps.
  6. Detector C uses a per-channel silence length (detectors.gap_pred_adaptive).
  7. Speed: pruning and the search run in parallel (fork workers), predictions can be
     assembled on the GPU (gpu_search.py); results are identical to a serial run.
     --deadline stops the search early (best TRAIN config kept) so the test always runs.

Typical Kaggle use (the notebook runs these as separate processes):
    python src/run_full_channels.py --mission M1_full --errors-only     # GPU: error arrays
    python src/run_full_channels.py --mission M1_full --stage search    # tune on TRAIN
    python src/run_full_channels.py --mission M1_full --stage test      # one TEST run
Outputs: results/<mission>_preregistered.json (chosen config) and <mission>_result.json.
"""
import os, sys, json, time, argparse, itertools, gc, hashlib
from collections import deque
import multiprocessing as mp
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MISSIONS, DATA, RESULTS, CACHE                              # noqa: E402
from dataset import build_arrays, timestamps                                   # noqa: E402
from model import GDNForecaster, RF, series_errors                             # noqa: E402
from detectors import (multi_span_z, global_multi_span_z, ewma,              # noqa: E402
                       zero_gap_margin, close_runs2d, open_runs2d,
                       channel_gap_cadence, gap_pred_adaptive, runs_of)
from esa_metrics import full_scoreboard, global_f05, build_labels_df, PRIMARY  # noqa: E402
from gpu_search import GpuSearch, order_stats_thresholds                       # noqa: E402
from telemanom_detector import telemanom_detect_multi                          # noqa: E402
import parallel_scoring                                                        # noqa: E402

DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
SPANS = [1, 3, 15, 60, 240]      # EWMA spans for Detector B

# Arrays shared with fork workers. Set in the parent just before a Pool starts; children
# inherit them copy-on-write, so multi-GB arrays are never pickled.
_G = {}


def _avail_gb():
    """Free RAM in GB (Linux /proc/meminfo, else psutil, else assume 8)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        return 8.0


def auto_workers(req, per_gb):
    """Number of worker processes. 'auto' = as many as CPU cores and free RAM allow
    (per_gb GB each, 4 GB kept back, at most 8); otherwise the requested number."""
    if req != "auto":
        return max(1, int(req))
    return max(1, min(os.cpu_count() or 1, int((_avail_gb() - 4.0) // per_gb), 8))


def target_channels(mk):
    """Channels that are both inputs and Target=YES in channels.csv -- the scored ones.
    ESA's infer_anomaly_types.py skips Target=NO channels, so they never get labels."""
    m = MISSIONS[mk]
    ch = pd.read_csv(f"{m['raw']}/channels.csv")
    keep = set(ch.loc[ch["Target"] == "YES", "Channel"])
    return [c for c in m["chs"] if c in keep]


def _prune_task(c):
    """Worker: Detector A for channel c at every p; returns run tables (small to send back)."""
    res = telemanom_detect_multi(_G["es"][:, c], _G["vals"][:, c], _G["ps"],
                                 num_to_ignore=_G["n_ignore"])
    return c, {p: runs_of(m > 0) for p, m in res.items()}


def prune_all(errs_s, vals, ps, n_ignore, workers=1, log=print):
    """Detector A for every channel and every p.

    Input : errs_s (T, C) EWMA(5) errors; vals (T, C) values; ps e.g. [0.3, 0.13, 0.05, None]
    Output: {p: (T, C) uint8}; p=None means "Detector A off" (all zeros)
    Channels run on `workers` fork processes.
    """
    T, C = errs_s.shape
    out = {p: np.zeros((T, C), np.uint8) for p in ps}
    real = [p for p in ps if p is not None]
    if not real:
        return out
    _G.update(es=errs_s, vals=vals, ps=real, n_ignore=n_ignore)

    def fill(c, runs):
        for p, r in runs.items():
            for a, b in r:
                out[p][a:b, c] = 1

    t0 = time.time()
    if workers > 1 and not torch.cuda.is_initialized():      # never fork a CUDA process
        with mp.get_context("fork").Pool(workers) as pool:
            for k, (c, runs) in enumerate(pool.imap_unordered(_prune_task, range(C)), 1):
                fill(c, runs)
                if k % 20 == 0:
                    log(f"      prune {k}/{C} channels ({time.time()-t0:.0f}s)")
    else:
        for c in range(C):
            fill(*_prune_task(c))
    for k in ("es", "vals"):
        _G.pop(k, None)                    # let the caller free these arrays
    return out


def thresholds_all(Z, qs, tmode):
    """Detector-B thresholds = quantiles of TRAIN z.

    Input : Z (T, C); qs e.g. [0.9999, 0.99999]; tmode "global" (one number) or "perchan"
    Output: {q: scalar} or {q: (C,) array}
    """
    if tmode == "global":
        v = np.quantile(Z, qs)
        return {q: np.float32(v[i]) for i, q in enumerate(qs)}
    v = np.quantile(Z, qs, axis=0)
    return {q: v[i].astype(np.float32) for i, q in enumerate(qs)}


AXES = ["p", "zmode", "tmode", "q", "close", "minlen", "gapL", "gapmult", "boost"]


def inert(kw, grid):
    """True if the config duplicates one already scored: with zmode="none" there is no
    Detector B, so q, tmode and boost have nothing to act on."""
    return kw["zmode"] == "none" and (
        kw["boost"] or kw["tmode"] == "perchan" or kw["q"] != grid["q"][0])


def _eval_cfg(kw):
    """One config -> TRAIN event-wise F0.5 (None if it predicts nothing). Reads inputs from _G."""
    g = _G
    if kw["zmode"] == "none":
        Z, thr = None, None
    else:
        Z, thr = g["ZTR"][kw["zmode"]], g["THR"][(kw["zmode"], kw["q"], kw["tmode"])]
    P = assemble(g["P_CACHE"][kw["p"]], Z, thr, g["gd"][(kw["gapL"], kw["gapmult"])],
                 g["tcw"], kw["boost"], kw["close"], kw["minlen"])
    if P.sum() == 0:
        return None
    r = global_f05(P[:, g["keep_idx"]], g["ts"], g["RAW"], g["SCORE_CHS"], labels_df=g["ldf"],
                   compress=True)
    return r["global_EW_F_0.50"]


def search(grid, mode="two-stage", log=print, workers=1, stop_at=None, scorer=None):
    """Grid search on the TRAIN period. Returns (best_config, best_F0.5, stats).

    mode "full"     : every combination (1,800 configs after removing duplicates).
    mode "two-stage": stage 1 pins close=60, minlen=3 and searches the other axes
                      (300 configs); stage 2 tries every close x minlen around the winner
                      (5 more). Measured on Kaggle: ~4.6 h for the 305 configs.
    Configs are scored in order and only a strictly better score wins, so the result is the
    same as a sequential run. stop_at (unix time) ends early, keeping the best so far.
    scorer: optional GPU scorer (gpu_scorer); default scores on CPU with _eval_cfg.
    """
    best = [None, -1.0]
    stats = dict(evaluated=0, truncated=False, workers=workers,
                 backend="gpu" if scorer else "cpu")

    def run(combos, phase):
        combos = [c for c in combos if not inert(c, grid)]
        n = len(combos)
        log(f"\n{phase}: {n} configs | {stats['backend'].upper()} assembly, "
            f"{workers} CPU worker(s) for ESA scoring")
        t0 = time.time()
        pool = (mp.get_context("fork").Pool(workers)
                if workers > 1 and not scorer and not torch.cuda.is_initialized() else None)
        results = None
        try:
            results = (scorer(combos) if scorer else
                       pool.imap(_eval_cfg, combos) if pool else map(_eval_cfg, combos))
            for j, (kw, f05) in enumerate(zip(combos, results), 1):
                stats["evaluated"] += 1
                if f05 is not None and f05 > best[1]:
                    best[0], best[1] = dict(kw), f05
                    log(f"  [{j}/{n}] {kw} -> F0.5={f05:.4f}  *")
                elif j % 25 == 0:
                    el = time.time() - t0
                    log(f"  [{j}/{n}] ... best {best[1]:.4f} | {el/60:.1f} min, "
                        f"~{el / j * (n - j) / 60:.0f} min left")
                if stop_at and time.time() > stop_at and j < n:
                    stats["truncated"] = True
                    log(f"  !! time budget reached after {j}/{n} configs -- keeping the "
                        f"best so far (TRAIN-selected) so the test still fits")
                    break
        finally:
            if pool:
                pool.terminate()
                pool.join()
            if scorer and hasattr(results, "close"):
                results.close()

    def product(g):
        return [dict(zip(AXES, v)) for v in itertools.product(*[g[a] for a in AXES])]

    if mode == "full":
        run(product(grid), "TRAIN tuning (full grid)")
        return best[0], best[1], stats

    run(product(dict(grid, close=[60], minlen=[3])),
        "TRAIN tuning stage 1 (close=60, minlen=3 pinned)")
    if best[0] is None or stats["truncated"]:
        return best[0], best[1], stats
    w = best[0]
    stage2 = [dict(w, close=c, minlen=m)
              for c in grid["close"] for m in grid["minlen"]
              if (c, m) != (w["close"], w["minlen"])]
    run(stage2, "TRAIN tuning stage 2 (close x minlen)")
    return best[0], best[1], stats


def gpu_scorer(gs, THR, pool, depth):
    """Build a scorer for search(): configs -> F0.5, in order.

    Each prediction is assembled on the GPU (gs = GpuSearch) and scored by ESA's metric on a
    CPU worker pool; up to `depth` scorings run at once. Identical predictions (common in a
    grid) are scored once, keyed by a hash of the prediction.
    """
    def scores(combos):
        pend, seen = deque(), {}
        def submit(kw):
            thr = None if kw["zmode"] == "none" else THR[(kw["zmode"], kw["q"], kw["tmode"])]
            pred = gs.pred1d(kw, thr)
            if pred is None:
                return None
            h = hashlib.blake2b(pred.tobytes(), digest_size=16).digest()
            if h not in seen:
                seen[h] = pool.apply_async(parallel_scoring.score1d, (pred,))
            return seen[h]
        for kw in combos:
            pend.append(submit(kw))
            if len(pend) > depth:
                r = pend.popleft()
                yield None if r is None else r.get()
        while pend:
            r = pend.popleft()
            yield None if r is None else r.get()
    return scores


def _z_task(job):
    """Worker: Detector-B z for one group of channels."""
    kind, c0, c1, block, chunk = job
    ec = _G["ec"][:, c0:c1]
    z = (multi_span_z(ec, SPANS, block, chunk=chunk) if kind == "Zl"
         else global_multi_span_z(ec, SPANS, chunk=chunk))
    return kind, c0, z


def z_stacks(ec, zkeys, block, chunk, workers, log=print):
    """Detector-B z arrays, computed in parallel over channel groups (same result as serial).

    Input : ec (T, C) gap-cleaned errors; zkeys subset of ("Zl" local, "Zg" whole-series)
    Output: {key: (T, C) float32}, first RF rows set to 0
    """
    T, C = ec.shape
    if isinstance(workers, str):                  # "auto" or a number from the CLI
        per = 40 * T * chunk / 1e9 + 0.5          # GB per worker (pandas EWM is float64)
        free = _avail_gb() - 4.0 - 4 * T * C * len(zkeys) / 1e9   # minus the outputs
        cap = (os.cpu_count() or 1) if workers == "auto" else int(workers)
        workers = max(1, min(cap, int(free // per)))
    out = {k: np.empty((T, C), np.float32) for k in zkeys}
    jobs = [(k, c0, min(c0 + chunk, C), block, chunk) for k in zkeys for c0 in range(0, C, chunk)]
    if workers > 1 and jobs and not torch.cuda.is_initialized():
        log(f"  z stacks: {len(jobs)} column groups on {workers} worker(s)")
        _G["ec"] = ec
        try:
            with mp.get_context("fork").Pool(workers) as pool:
                for k, c0, z in pool.imap_unordered(_z_task, jobs):
                    out[k][:, c0:c0 + z.shape[1]] = z
        finally:
            _G.pop("ec", None)
    else:
        for k in zkeys:
            out[k] = (multi_span_z(ec, SPANS, block, chunk=chunk) if k == "Zl"
                      else global_multi_span_z(ec, SPANS, chunk=chunk))
    for k in zkeys:
        out[k][:RF] = 0
    return out


def assemble(prune_P, Z, thr, gapdet, tcw, boost, close, minlen):
    """OR the four detectors per channel, then merge short gaps and drop short alarms.

    Input : prune_P (T, C) Detector A; Z (T, C) + thr (scalar or (C,)) Detector B;
            gapdet (T, C) Detector C; tcw (T,) telecommand window; boost (Detector D)
    Output: (T, C) uint8
    """
    P = prune_P.copy()                                                     # A
    if thr is not None:
        P = np.maximum(P, (Z > thr).astype(np.uint8))                      # B
        if boost:
            P = np.maximum(P, ((Z > thr * boost) & tcw[:, None]).astype(np.uint8))   # D
    if gapdet is not None:
        P = np.maximum(P, gapdet)                                          # C
    return open_runs2d(close_runs2d(P, close), minlen)                     # cleanup


def phase(mk, split, model, mu, sd, block, chunk, cache=True, zkeys=("Zl", "Zg"),
          workers=1):
    """Build every detector input for one split.

    Output: dict with vals (T, C), cnt (T, C), es5 (EWMA-5 errors), tcw (T,), ts, mu, sd,
            n_ch and the requested z arrays (Zl / Zg).
    The gap-cleaned errors and the values are cached as .npy in CACHE, so a re-run after a
    Kaggle timeout skips the GPU error pass.
    """
    t0 = time.time()
    ec_f, va_f = f"{CACHE}/{mk}_{split}_ec.npy", f"{CACHE}/{mk}_{split}_valscache.npy"
    f = np.load(f"{DATA}/{mk}_{split}_feats.npz")
    cnt, tcdt = f["count"], f["tcdt"]

    if cache and os.path.exists(ec_f) and os.path.exists(va_f):
        ec, vals = np.load(ec_f), np.load(va_f)
        n_ch = vals.shape[1]
        if mu is None:                       # normalisation stats still needed for test
            _, mu, sd, _, _ = build_arrays(mk, split)
        print(f"  [{split}] reusing cached errors ({time.time()-t0:.0f}s)", flush=True)
    else:
        X, mu, sd, n_ch, n_tc = build_arrays(mk, split, mu, sd)
        err = series_errors(model, X, DEV)
        vals = np.ascontiguousarray(X[:, :n_ch, 0])   # Detector A needs the channel values
        del X                                         # X is ~7.7 GB at full scale
        gc.collect()
        print(f"  [{split}] errors done, X freed ({time.time()-t0:.0f}s)", flush=True)
        ec = zero_gap_margin(err, (cnt == 0).all(1), 30)       # step 1
        del err
        gc.collect()
        if cache:
            os.makedirs(CACHE, exist_ok=True)
            np.save(ec_f, ec); np.save(va_f, vals)

    d = dict(vals=vals, cnt=cnt, es5=ewma(ec, 5),
             tcw=(tcdt.min(1) <= 7200.0), ts=timestamps(mk, split),    # D: 2 h window
             n_ch=n_ch, mu=mu, sd=sd)
    d.update(z_stacks(ec, zkeys, block, chunk, workers))
    del ec
    del tcdt
    gc.collect()
    print(f"  [{split}] z stacks built ({time.time()-t0:.0f}s)", flush=True)
    return d


def zmap(d, sl, sg):
    """The three Detector-B variants: local z, whole-series z, or the max of both after
    scaling each by its own TRAIN 0.999 quantile (sl, sg)."""
    return {"local": d["Zl"], "global": d["Zg"],
            "max": np.maximum(d["Zl"] / sl, d["Zg"] / sg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="M1_full", choices=["M1_full", "M2_full"])
    ap.add_argument("--ckpt", default=None, help="trained model; default results/<mission>_model.pt")
    ap.add_argument("--block", type=int, default=2880, help="Detector-B baseline block (1 day)")
    ap.add_argument("--chunk", type=int, default=16, help="channels per z-score chunk")
    ap.add_argument("--tag", default="", help="optional suffix for the output file names")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 9-config grid that touches every branch; checks the pipeline "
                         "runs end to end -- its scores are NOT results")
    ap.add_argument("--search", choices=["two-stage", "full"], default="two-stage",
                    help="two-stage = 305 configs (~4.6 h on Kaggle); full = 1,800 configs")
    ap.add_argument("--workers", default="auto",
                    help="worker processes for pruning, z and the search; 'auto' sizes to "
                         "cores and free RAM. Results do not depend on it.")
    ap.add_argument("--deadline", type=float, default=0.0,
                    help="unix time this script must finish by (0 = none). The search stops "
                         "early, keeping its best TRAIN config, to leave --test-reserve-min.")
    ap.add_argument("--test-reserve-min", type=float, default=90.0, dest="test_reserve_min")
    ap.add_argument("--reuse-prereg", action="store_true", dest="reuse_prereg",
                    help="if a pre-registration exists (TRAIN choice whose test never ran), "
                         "skip the search and go straight to the test")
    ap.add_argument("--stage", choices=["all", "search", "test"], default="all",
                    help="search: tune on TRAIN, save the config, exit. test: run the single "
                         "TEST evaluation of a saved config.")
    ap.add_argument("--detect-device", choices=["auto", "cuda", "cpu", "torch-cpu"],
                    default="auto", dest="detect_device",
                    help="where the TRAIN search assembles predictions (auto: cuda if present)")
    ap.add_argument("--errors-only", action="store_true", dest="errors_only",
                    help="GPU step: compute + cache the train/test error arrays, then exit")
    args = ap.parse_args()
    mk = args.mission
    m = MISSIONS[mk]
    RAW = m["raw"]
    SCORE_CHS = target_channels(mk)
    all_chs = m["chs"]
    keep_idx = np.array([all_chs.index(c) for c in SCORE_CHS])     # scored columns
    t0 = time.time()
    is_m1 = mk.startswith("M1")               # Detector C is used on Mission 1 only
    ckpt = args.ckpt or f"{RESULTS}/{mk}_model.pt"
    stem = f"{RESULTS}/{mk}" + (f"_{args.tag}" if args.tag else "")
    prereg_f, result_f = f"{stem}_preregistered.json", f"{stem}_result.json"
    os.makedirs(RESULTS, exist_ok=True)

    print(f"[{mk}] dev={DEV}  channels={len(all_chs)} (inputs)  "
          f"{len(SCORE_CHS)} Target=YES (scored)  cpus={os.cpu_count()}  "
          f"free RAM {_avail_gb():.1f} GB", flush=True)
    if DEV == "cuda":
        torch.backends.cudnn.benchmark = True
    log = lambda msg: print(msg, flush=True)

    # ---------------- model (only needed if the error arrays are not cached) ----------------
    n_ch = len(all_chs)
    n_tc = int(np.load(f"{DATA}/{mk}_train_feats.npz")["tcdt"].shape[1])
    cached = all(os.path.exists(f"{CACHE}/{mk}_{s}_{k}.npy")
                 for s in ("train", "test") for k in ("ec", "valscache"))
    model = None
    if not cached:
        model = GDNForecaster(n_ch, n_tc).to(DEV)
        model.load_state_dict(torch.load(ckpt, map_location=DEV))
        model.eval()
        print(f"[{mk}] model n_ch={n_ch} n_tc={n_tc} topk={model.topk}", flush=True)
    else:
        print(f"[{mk}] all error arrays cached in {CACHE} -- model not loaded", flush=True)
    if args.errors_only:
        mu_ = sd_ = None
        for split in ([] if cached else ["train", "test"]):
            X, m_, s_, _, _ = build_arrays(mk, split, mu_, sd_)   # test uses train stats
            if split == "train":
                mu_, sd_ = m_, s_
            err = series_errors(model, X, DEV)
            cnt = np.load(f"{DATA}/{mk}_{split}_feats.npz")["count"]
            os.makedirs(CACHE, exist_ok=True)
            np.save(f"{CACHE}/{mk}_{split}_valscache.npy", np.ascontiguousarray(X[:, :n_ch, 0]))
            del X
            np.save(f"{CACHE}/{mk}_{split}_ec.npy",                     # same as phase()
                    zero_gap_margin(err, (cnt == 0).all(1), 30))
            del err, cnt
            gc.collect()
            print(f"  [{split}] error arrays cached ({time.time()-t0:.0f}s)", flush=True)
        print(f"errors-only: done ({time.time()-t0:.0f}s)", flush=True)
        return

    # ---------------- TRAIN phase: tune, pre-register, then free ----------------
    want_pre = args.reuse_prereg or args.stage == "test"
    pre = json.load(open(prereg_f)) if (want_pre and os.path.exists(prereg_f)) else None
    if args.stage == "test" and pre is None:
        raise SystemExit(f"--stage test needs {prereg_f} (run --stage search first)")
    if pre and args.stage == "search":
        log(f"  pre-registration already exists ({prereg_f}) -- search skipped")
        return
    if pre and not (pre.get("mission") == mk and bool(pre.get("smoke")) == args.smoke
                    and (pre["config"]["zmode"] != "max" or "sl" in pre)):
        log(f"  {prereg_f} does not match this run -- tuning again")
        pre = None
    mu = sd = None
    if pre:
        # reuse a TRAIN-only selection whose test never completed: same frozen config
        cfg, best_f05, stats = pre["config"], pre["train_f05"], pre.get("search", {})
        best = (cfg, best_f05)
        thr = (None if cfg["zmode"] == "none" else
               np.float32(pre["thr"][0]) if cfg["tmode"] == "global"
               else np.asarray(pre["thr"], np.float32))
        cadence = None if pre.get("cadence") is None else np.asarray(pre["cadence"], np.float32)
        sl, sg = pre.get("sl", 1.0), pre.get("sg", 1.0)
        log(f"\n>>> REUSING PRE-REGISTERED config (chosen on TRAIN, F0.5={best_f05:.4f}, written "
            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(prereg_f)))}): {cfg}")
    else:
        tr = phase(mk, "train", model, None, None, args.block, args.chunk, workers=args.workers)
        mu, sd = tr["mu"], tr["sd"]
        sl = float(np.quantile(tr["Zl"], 0.999)) or 1.0
        sg = float(np.quantile(tr["Zg"], 0.999)) or 1.0
        ZTR = zmap(tr, sl, sg)

        # Grid axes. Order matters only if --deadline cuts the search short: the first
        # config is the lightweight winner plus the full-scale gap fix (gapmult=6).
        grid = dict(p=[0.3, 0.13, 0.05, None], zmode=["local", "global", "max", "none"],
                    tmode=["global", "perchan"], q=[0.9999, 0.99999],
                    close=[0, 60, 240], minlen=[1, 3], gapL=[10],
                    gapmult=[6.0, 3.0, None], boost=[0.6, None])
        if args.smoke:
            print("!! --smoke: 9-config grid, scores below are NOT results", flush=True)
            grid = dict(p=[None, 0.3], zmode=["local", "global"], tmode=["perchan"],
                        q=[0.9999], close=[0, 60, 240], minlen=[1, 3], gapL=[10],
                        gapmult=[6.0], boost=[0.6])
        T, C = tr["es5"].shape
        w_prune = auto_workers(args.workers, 100 * T / 1e9 + 0.3)
        log(f"  pruning {C} channels x p={[p for p in grid['p'] if p is not None]} "
            f"on {w_prune} worker(s)")
        P_CACHE = prune_all(tr["es5"], tr["vals"], grid["p"], RF, w_prune, log)    # A
        for p in grid["p"]:
            log(f"  prune p={p} rows={int(P_CACHE[p].max(1).sum()):,} ({time.time()-t0:.0f}s)")
        del tr["es5"], tr["vals"]
        gc.collect()

        # Detector C: cadence fitted on TRAIN, carried unchanged to TEST
        cadence = channel_gap_cadence(tr["cnt"]) if is_m1 else None
        gd_tr = {(L, mu_): (gap_pred_adaptive(tr["cnt"], L, cadence, mu_) if is_m1 else None)
                 for L in grid["gapL"] for mu_ in grid["gapmult"]}
        ldf_tr = build_labels_df(f"{RAW}/labels.csv", f"{RAW}/anomaly_types.csv",
                                 pd.to_datetime(pd.Series(tr["ts"])))

        # Detector B thresholds for every (zmode, q, tmode), computed once before forking
        sdev = ("cuda" if args.detect_device in ("auto", "cuda") and torch.cuda.is_available()
                else "torch-cpu" if args.detect_device == "torch-cpu" else "cpu")
        tdev = "cuda" if sdev == "cuda" else "cpu"     # torch-cpu: GPU code path, for tests
        THR = {}
        for zm in [z for z in grid["zmode"] if z != "none"]:
            for tm in grid["tmode"]:
                try:
                    vals_ = (order_stats_thresholds(ZTR[zm], grid["q"], tm, tdev)
                             if sdev != "cpu" else thresholds_all(ZTR[zm], grid["q"], tm))
                except RuntimeError as e:                     # e.g. CUDA out of memory
                    log(f"  GPU thresholds failed ({str(e)[:80]}) -- numpy fallback")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    vals_ = thresholds_all(ZTR[zm], grid["q"], tm)
                for q, v in vals_.items():
                    THR[(zm, q, tm)] = v
        log(f"  {len(THR)} thresholds precomputed on {sdev} ({time.time()-t0:.0f}s)")

        # inputs the search workers read (inherited through fork)
        _G.update(ZTR=ZTR, THR=THR, P_CACHE=P_CACHE, gd=gd_tr, tcw=tr["tcw"], ts=tr["ts"],
                  RAW=RAW, SCORE_CHS=SCORE_CHS, keep_idx=keep_idx, ldf=ldf_tr)
        w_search = auto_workers(args.workers, 6.5 * T * C / 1e9 + 0.3)
        stop_at = args.deadline - args.test_reserve_min * 60 if args.deadline else None
        if stop_at:
            log(f"  search must end by {time.strftime('%H:%M', time.localtime(stop_at))} "
                f"({(stop_at - time.time()) / 60:.0f} min from now)")
        gs = scorer = spool = None
        if sdev != "cpu":                        # GPU assembly + CPU scoring pool
            try:
                gs = GpuSearch(ZTR, P_CACHE, gd_tr, tr["tcw"], keep_idx, tdev, log=log)
                w_search = auto_workers(args.workers, 3.0)
                spool = parallel_scoring.make_pool(w_search, tr["ts"], RAW, SCORE_CHS, ldf_tr)
                scorer = gpu_scorer(gs, THR, spool, depth=2 * w_search)
            except RuntimeError as e:                         # CUDA out of memory etc.
                log(f"  GPU search unavailable ({str(e)[:100]}) -- CPU search")
                gs = scorer = None
                if spool:
                    spool.terminate()
                    spool = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        try:
            try:
                cfg, best_f05, stats = search(grid, mode=args.search, log=log, workers=w_search,
                                              stop_at=stop_at, scorer=scorer)
            except RuntimeError as e:
                if scorer is None:
                    raise
                log(f"  GPU search failed mid-way ({str(e)[:100]}) -- restarting on CPU")
                gs = scorer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                w_search = auto_workers(args.workers, 6.5 * T * C / 1e9 + 0.3)
                cfg, best_f05, stats = search(grid, mode=args.search, log=log, workers=w_search,
                                              stop_at=stop_at)
        finally:
            if spool:
                spool.terminate()
                spool.join()
            del gs, scorer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _G.clear()
        if cfg is None:
            raise SystemExit("no config produced a single positive prediction on TRAIN -- "
                             "nothing to pre-register; check the error pass and the gap features")
        best = (cfg, best_f05)
        cfg = best[0]
        print(f"\n>>> PRE-REGISTERED (train F0.5={best[1]:.4f}): {cfg}", flush=True)

        # save the chosen config with its TRAIN-fitted threshold values (never refitted on test)
        thr = None if cfg["zmode"] == "none" else THR[(cfg["zmode"], cfg["q"], cfg["tmode"])]
        json.dump({"mission": mk, "config": cfg, "train_f05": best[1], "spans": SPANS,
                   "block": args.block, "scored_channels": SCORE_CHS,
                   "n_input_channels": len(all_chs), "smoke": args.smoke, "search": stats,
                   "thr": None if thr is None else np.atleast_1d(thr).tolist(),
                   "sl": sl, "sg": sg,
                   "cadence": None if cadence is None else cadence.tolist()},
                  open(prereg_f, "w"), indent=1)

        del tr, ZTR, P_CACHE, gd_tr, ldf_tr, THR
        gc.collect()
        print(f"[{mk}] train phase freed ({time.time()-t0:.0f}s)", flush=True)
        if args.stage == "search":
            log(f"search stage done ({time.time()-t0:.0f}s)")
            return

    # ---------------- TEST phase: one evaluation of the frozen config ----------------
    if mu is None and not all(os.path.exists(f"{CACHE}/{mk}_test_{k}.npy")
                              for k in ("ec", "valscache")):
        _, mu, sd, _, _ = build_arrays(mk, "train")        # test is normalised with TRAIN stats
    zm = cfg["zmode"]
    zkeys = {"none": (), "local": ("Zl",), "global": ("Zg",), "max": ("Zl", "Zg")}[zm]
    te = phase(mk, "test", model, mu, sd, args.block, args.chunk, zkeys=zkeys,
               workers=args.workers)
    Z_te = (None if zm == "none" else te["Zl"] if zm == "local" else te["Zg"] if zm == "global"
            else np.maximum(te["Zl"] / sl, te["Zg"] / sg))
    for k in zkeys:
        del te[k]
    T, C = te["es5"].shape
    P_prune = prune_all(te["es5"], te["vals"], [cfg["p"]], RF,
                        auto_workers(args.workers, 100 * T / 1e9 + 0.3), log)[cfg["p"]]
    del te["es5"], te["vals"]
    gc.collect()
    gd_te = (gap_pred_adaptive(te["cnt"], cfg["gapL"], cadence, cfg["gapmult"])
             if is_m1 else None)
    P_te = assemble(P_prune, Z_te, thr, gd_te, te["tcw"], cfg["boost"],
                    cfg["close"], cfg["minlen"])
    P_te = np.ascontiguousarray(P_te[:, keep_idx])     # scored (Target=YES) channels only
    del P_prune, Z_te, gd_te, te["cnt"]
    gc.collect()
    log(f"  test predictions assembled, scoring ({time.time()-t0:.0f}s, free RAM {_avail_gb():.1f} GB)")

    # compress=True: score only the change points (same scores, far less memory)
    sb = full_scoreboard(P_te, te["ts"], RAW, SCORE_CHS, compress=True)
    print(f"\n===== {mk} TEST -- ALL ESA ASPECTS ({len(SCORE_CHS)} scored channels) =====",
          flush=True)
    for k in PRIMARY:
        if k in sb:
            print(f"  {k:34s} {sb[k]:.4f}", flush=True)
    for k, v in sb.items():
        if k not in PRIMARY:
            print(f"  {k:34s} {v}", flush=True)
    json.dump({"mission": mk, "config": cfg, "train_f05": best[1], "spans": SPANS,
               "block": args.block, "scored_channels": SCORE_CHS,
               "n_input_channels": len(all_chs), "smoke": args.smoke, "search": stats,
               "reused_preregistration": bool(pre), "test": sb},
              open(result_f, "w"), indent=1)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
