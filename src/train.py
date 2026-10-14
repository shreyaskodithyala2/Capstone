"""Train the forecasting model (GDNForecaster) on one mission's training split.

Self-supervised: the target is simply the next value of every channel, so no anomaly
labels are used here. Output is one checkpoint, results/<mission>_model.pt.

    python src/train.py --mission M1                       # lightweight, ~40 s/epoch on MPS
    python src/train.py --mission M1_full --val-from 2006-10-01   # full set (Kaggle GPU)

Training loop: random 512-step segments, batch 24, 300 batches per epoch, Adam 1e-3,
MSE loss, up to 30 epochs, early stop after 4 epochs without validation improvement.
Validation = the last part of the training period (chronological, never shuffled).
On CUDA it can also use 2 GPUs (DataParallel) and fp16 mixed precision when faster.
"""
import os, sys, time, argparse
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MISSIONS, RESULTS
from dataset import build_arrays, timestamps
from model import GDNForecaster, RF

torch.manual_seed(0); np.random.seed(0)
DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")


def sample_batch(X, starts, seg, n_ch, rng, bs):
    """Draw `bs` random training segments.

    Input : X (T, N, 3), starts = allowed segment start rows, seg = segment length
    Output: x (bs, N, seg, 3) inputs, y (bs, seg-RF, n_ch) next-step targets
    Example: seg=512 -> each segment gives 512-253 = 259 supervised predictions.
    """
    ss = rng.choice(starts, bs, replace=False)
    xs = np.stack([np.transpose(X[s:s + seg], (1, 0, 2)) for s in ss])   # (B, N, L, F)
    ys = np.stack([X[s + RF:s + seg, :n_ch, 0] for s in ss])             # (B, L-RF, n_ch)
    return torch.from_numpy(xs), torch.from_numpy(ys)


class _Segments(torch.nn.Module):
    """Wraps the model so forward() == forward_segments(); nn.DataParallel needs a forward().
    autocast is entered here because DataParallel runs each GPU replica in its own thread."""
    def __init__(self, m):
        super().__init__()
        self.m, self.amp = m, False

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            return self.m.forward_segments(x).float()


def _prefetch(X, starts, seg, n_ch, rng, bs, n, pin):
    """Yield `n` batches, sliced by a background thread while the GPU trains on the current
    one. A single thread draws from `rng` in order, so the batches are the same as without it."""
    import queue, threading
    q = queue.Queue(maxsize=3)

    def work():
        for _ in range(n):
            x, y = sample_batch(X, starts, seg, n_ch, rng, bs)
            q.put((x.pin_memory(), y.pin_memory()) if pin else (x, y))
    threading.Thread(target=work, daemon=True).start()
    for _ in range(n):
        yield q.get()


def _time_steps(net, opt, X, starts, args, n_ch, amp, steps=6):
    """Milliseconds per training step with AMP on or off, measured on throw-away copies
    of the model and optimiser (the real run is not affected)."""
    import copy
    net, opt2 = copy.deepcopy(net), None
    params = [p for p in net.parameters()]
    opt2 = torch.optim.Adam(params, lr=args.lr)
    dpnet = (torch.nn.DataParallel(net) if DEV == "cuda" and args.dp == "auto"
             and torch.cuda.device_count() > 1 else net)
    net.amp = amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.default_rng(123)
    lossf = torch.nn.MSELoss()
    for i in range(steps + 2):
        if i == 2:                                      # skip 2 warm-up steps
            torch.cuda.synchronize(); t0 = time.time()
        x, y = sample_batch(X, starts, args.seg, n_ch, rng, args.bs)
        opt2.zero_grad()
        loss = lossf(dpnet(x.to(DEV)), y.to(DEV))
        scaler.scale(loss).backward(); scaler.step(opt2); scaler.update()
    torch.cuda.synchronize()
    return (time.time() - t0) / steps * 1000


def train_model(X, n_ch, n_tc, args, vsplit):
    """Train a fresh GDNForecaster and return it with the best-validation weights loaded.

    Input : X (T, N, 3) training-split tensor; vsplit = first validation row
            (rows < vsplit train, rows >= vsplit validate)
    Output: trained model (on DEV)
    """
    model = GDNForecaster(n_ch, n_tc, topk=args.topk).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.MSELoss()
    rng = np.random.default_rng(0)
    tr_starts = np.arange(0, vsplit - args.seg - 1)
    if args.val_from:      # 80 validation segments spread over the whole validation window
        va_starts = np.linspace(vsplit, len(X) - args.seg - 1, 80).astype(int)
    else:                  # first 80 consecutive segments after vsplit (lightweight runs)
        va_starts = np.arange(vsplit, len(X) - args.seg - 1, args.seg)[:80]

    net = _Segments(model)
    fwd = net if DEV == "cuda" else model.forward_segments
    # Two GPUs (Kaggle "T4 x2"): DataParallel splits each batch of 24 into 12 + 12.
    if DEV == "cuda" and args.dp == "auto" and torch.cuda.device_count() > 1:
        dp = torch.nn.DataParallel(net)
        try:
            x, y = sample_batch(X, tr_starts, args.seg, n_ch, np.random.default_rng(1), 2)
            with torch.no_grad():
                dp(x.to(DEV))                           # smoke test before committing to it
            fwd = dp
            print(f"  DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        except RuntimeError as e:
            print(f"  DataParallel unavailable ({str(e)[:100]}) -- single GPU", flush=True)

    # fp16 mixed precision only if it is measurably faster on this GPU
    use_amp = False
    if DEV == "cuda" and args.amp != "off":
        try:
            t32 = _time_steps(net, opt, X, tr_starts, args, n_ch, False)
            t16 = _time_steps(net, opt, X, tr_starts, args, n_ch, True)
            use_amp = args.amp == "on" or t16 < 0.9 * t32
            print(f"  step time fp32 {t32:.0f} ms | AMP fp16 {t16:.0f} ms -> "
                  f"{'AMP' if use_amp else 'fp32'}", flush=True)
        except RuntimeError as e:
            print(f"  AMP check failed ({str(e)[:100]}) -- fp32", flush=True)
        torch.cuda.empty_cache()
    net.amp = use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    nonfinite = 0

    best, bad, best_state, ep_dur = 1e9, 0, None, 0.0
    for ep in range(args.epochs):
        if args.stop_by and ep and time.time() + ep_dur > args.stop_by:   # time budget
            print(f"  time budget: stopping before epoch {ep+1} (best val {best:.5f} kept)",
                  flush=True)
            break
        t_ep = time.time()
        model.train(); tl = torch.zeros((), device=DEV)
        for x, y in _prefetch(X, tr_starts, args.seg, n_ch, rng, args.bs, args.batches,
                              DEV == "cuda"):
            opt.zero_grad()
            loss = lossf(fwd(x.to(DEV, non_blocking=True)), y.to(DEV, non_blocking=True))
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                if not torch.isfinite(loss):          # the scaler skipped this step
                    nonfinite += 1
                    if nonfinite >= 20:               # fp16 keeps overflowing -> give up on it
                        print("  AMP produced 20 non-finite losses -- switching to fp32",
                              flush=True)
                        use_amp = net.amp = False
                        scaler = torch.amp.GradScaler("cuda", enabled=False)
                    continue
                nonfinite = 0
            else:
                loss.backward(); opt.step()
            tl += loss.detach()
        tl = float(tl)
        # ---- validation (always fp32) ----
        model.eval(); vl = 0.0
        amp_was, net.amp = net.amp, False
        with torch.no_grad():
            for i in range(0, len(va_starts), args.bs):
                ss = va_starts[i:i + args.bs]
                xs = np.stack([np.transpose(X[s:s + args.seg], (1, 0, 2)) for s in ss])
                ys = np.stack([X[s + RF:s + args.seg, :n_ch, 0] for s in ss])
                vl += lossf(fwd(torch.from_numpy(xs).to(DEV)),
                            torch.from_numpy(ys).to(DEV)).item() * len(ss)
        net.amp = amp_was
        vl /= len(va_starts)
        ep_dur = time.time() - t_ep
        print(f"  ep{ep+1}: train={tl/args.batches:.5f} val={vl:.5f} "
              f"({ep_dur:.0f}s, {ep_dur/args.batches*1000:.0f} ms/batch)", flush=True)
        if not np.isfinite(vl):
            print("  validation loss is not finite -- stopping, best weights kept", flush=True)
            break
        # ---- early stopping: keep the weights with the lowest validation loss ----
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stop", flush=True); break
    if best_state is None:
        raise SystemExit("training produced no finite validation loss -- no checkpoint")
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="M1", choices=list(MISSIONS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batches", type=int, default=300, help="batches per epoch")
    ap.add_argument("--bs", type=int, default=24, help="segments per batch")
    ap.add_argument("--seg", type=int, default=512, help="segment length in steps")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4, help="early-stopping patience (epochs)")
    ap.add_argument("--topk", type=int, default=None, help="graph neighbours; default min(n_ch-1, 10)")
    ap.add_argument("--ckpt", default=None, help="output path; default results/<mission>_model.pt")
    ap.add_argument("--val-from", default=None, dest="val_from",
                    help="validation = training rows from this date on (ESA-ADB holds out the "
                         "last 3 months, 2006-10-01 for M1). Unset = last 15%% of the train split.")
    ap.add_argument("--stop-by", type=float, default=0.0, dest="stop_by",
                    help="unix time: start no epoch that would end after it (best kept)")
    ap.add_argument("--amp", choices=["auto", "on", "off"], default="auto",
                    help="fp16 mixed precision on CUDA; auto = only if measured faster")
    ap.add_argument("--dp", choices=["auto", "off"], default="auto",
                    help="DataParallel across GPUs when more than one is visible")
    args = ap.parse_args()
    mk = args.mission; t0 = time.time()
    ckpt = args.ckpt or f"{RESULTS}/{mk}_model.pt"
    os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)

    if DEV == "cuda":
        torch.backends.cudnn.benchmark = True      # fixed input shapes -> pick fastest kernels once
    Xtr, mu, sd, n_ch, n_tc = build_arrays(mk, "train")
    if args.val_from:
        ts_tr = timestamps(mk, "train").to_numpy().astype("datetime64[ns]")
        vsplit = int(np.searchsorted(ts_tr, np.datetime64(pd.Timestamp(args.val_from))))
        print(f"validation from {args.val_from}: rows {vsplit:,}..{len(Xtr):,} "
              f"({len(Xtr) - vsplit:,} rows)", flush=True)
    else:
        vsplit = int(0.85 * len(Xtr))
    print(f"{mk}: T_train={len(Xtr):,} n_ch={n_ch} n_tc={n_tc} dev={DEV}", flush=True)

    model = train_model(Xtr, n_ch, n_tc, args, vsplit)
    torch.save(model.state_dict(), ckpt)
    print(f"saved {ckpt}", flush=True)
    print(f"trained in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
