"""GPU (PyTorch) versions of the slowest parts of the full-set TRAIN search.

Only comparisons, ORs and integer index arithmetic happen here, so results are bit-identical
to the numpy code in run_full_channels.assemble and detectors.close_runs2d / open_runs2d.
ESA's metric itself stays on the CPU (parallel_scoring.py).
"""
import numpy as np
import torch


def morph_rows(P, close, minlen):
    """close_runs + open_runs for many channels at once, on the GPU.

    Input : P (k, T) bool tensor, one channel per row, time along dim 1
    Output: (k, T) bool tensor
    Method: find every run's [start, end) from the 0/1 edges, merge runs whose gap is
            <= close, drop runs shorter than minlen, then rebuild the mask with a cumsum.
    """
    k, T = P.shape
    if close <= 0 and minlen <= 1:
        return P
    Q = torch.zeros((k, T + 2), dtype=torch.bool, device=P.device)
    Q[:, 1:-1] = P                                     # pad with 0 on both sides
    nz = torch.nonzero(Q[:, 1:] != Q[:, :-1])          # (2R, 2): (row, col) of every edge
    if nz.shape[0] == 0:
        return P
    row, s, e = nz[0::2, 0], nz[0::2, 1], nz[1::2, 1]  # edges alternate: start, end, ...
    if close > 0 and len(s) > 1:
        nxt = (row[:-1] == row[1:]) & ((s[1:] - e[:-1]) <= close)   # merge run i with i+1
        first = torch.ones_like(s, dtype=torch.bool)
        first[1:] = ~nxt
        last = torch.ones_like(s, dtype=torch.bool)
        last[:-1] = ~nxt
        row, s, e = row[first], s[first], e[last]
    if minlen > 1:
        keep = (e - s) >= minlen
        row, s, e = row[keep], s[keep], e[keep]
    W = T + 1
    delta = torch.zeros(k * W, dtype=torch.int32, device=P.device)
    delta.index_add_(0, row * W + s, torch.ones_like(s, dtype=torch.int32))        # +1 at start
    delta.index_add_(0, row * W + e, torch.full_like(e, -1, dtype=torch.int32))    # -1 at end
    return (torch.cumsum(delta, 0, dtype=torch.int32) > 0).view(k, W)[:, :T]


class GpuSearch:
    """Keeps every TRAIN detector input on the GPU and turns one search config into the
    1-D global prediction (OR over the scored channels) that ESA's metric scores.

    Input : ZTR {zmode: (T, C) z}, P_CACHE {p: (T, C) Detector A}, gd {(L, mult): (T, C)
            Detector C}, tcw (T,) telecommand window, keep_idx = scored channel columns
    """

    def __init__(self, ZTR, P_CACHE, gd, tcw, keep_idx, device="cuda", chunk=32, log=print):
        self.dev, self.chunk = torch.device(device), chunk

        def to_rows(a, dt=None):                       # (T, C) numpy -> (C, T) device tensor
            t = torch.from_numpy(np.ascontiguousarray(a)).to(self.dev, dt)
            out = t.t().contiguous()
            del t
            return out
        self.Z = {k: to_rows(v) for k, v in ZTR.items()}                   # (C, T) float32
        self.P = {p: (to_rows(v, torch.bool) if v.any() else None) for p, v in P_CACHE.items()}
        self.gd = {k: (None if v is None else to_rows(v, torch.bool)) for k, v in gd.items()}
        self.tcw = torch.from_numpy(np.ascontiguousarray(tcw)).to(self.dev, torch.bool).unsqueeze(0)
        self.C, self.T = next(iter(self.Z.values())).shape
        keep = set(int(i) for i in keep_idx)
        self.keep = [int(i) for i in keep_idx]
        self.rest = [c for c in range(self.C) if c not in keep]   # unscored channels
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()
            log(f"  GPU search inputs on {torch.cuda.get_device_name(self.dev)}: "
                f"{torch.cuda.memory_allocated(self.dev)/1e9:.1f} GB")

    def _thr(self, thr, cols):
        """Threshold as a (k, 1) column (per-channel) or a 0-d tensor (global)."""
        if np.ndim(thr) == 0:
            return torch.tensor(np.float32(thr), device=self.dev)
        return torch.from_numpy(np.asarray(thr, np.float32)[cols]).to(self.dev).unsqueeze(1)

    def _cols(self, cols, kw, thr):
        """Detectors A-D + cleanup for the given channels, OR-ed together -> (T,) bool."""
        out = torch.zeros(self.T, dtype=torch.bool, device=self.dev)
        Z = None if kw["zmode"] == "none" else self.Z[kw["zmode"]]
        Pp, gd = self.P[kw["p"]], self.gd[(kw["gapL"], kw["gapmult"])]
        for i in range(0, len(cols), self.chunk):
            cs = cols[i:i + self.chunk]
            ci = torch.tensor(cs, device=self.dev)
            P = (Pp.index_select(0, ci) if Pp is not None                          # A
                 else torch.zeros((len(cs), self.T), dtype=torch.bool, device=self.dev))
            if Z is not None:
                Zc = Z.index_select(0, ci)
                P |= Zc > self._thr(thr, cs)                                       # B
                if kw["boost"]:
                    # thr * boost computed exactly as numpy does in assemble()          D
                    P |= (Zc > self._thr(np.float32(thr * kw["boost"]) if np.ndim(thr) == 0
                                         else (thr * kw["boost"]).astype(np.float32), cs)) \
                        & self.tcw
                del Zc
            if gd is not None:
                P |= gd.index_select(0, ci)                                        # C
            out |= morph_rows(P, kw["close"], kw["minlen"]).any(0)                 # cleanup + OR
        return out

    def pred1d(self, kw, thr):
        """Global prediction over the scored channels as uint8 (T,) numpy, or None when no
        channel at all (scored or not) fires -- the same skip rule as the CPU path."""
        g = self._cols(self.keep, kw, thr)
        if not bool(g.any()) and not (self.rest and bool(self._cols(self.rest, kw, thr).any())):
            return None
        return g.to(torch.uint8).cpu().numpy()


def order_stats_thresholds(Z, qs, tmode, device="cuda"):
    """Same output as run_full_channels.thresholds_all, with the slow part on the GPU.

    np.quantile only reads the two sorted values around each target rank. Those are found
    with torch.kthvalue (selection, no full sort); numpy then interpolates on a small
    stand-in array holding the same values at the same ranks, so the result is exact.
    Output: {q: scalar} (tmode "global") or {q: (C,) array} (tmode "perchan")
    """
    dev = torch.device(device)
    if tmode == "global":
        cols = [np.ravel(Z)]
    else:
        cols = [Z[:, c] for c in range(Z.shape[1])]
    out = {q: [] for q in qs}
    for col in cols:
        n = len(col)
        x = torch.from_numpy(np.ascontiguousarray(col)).to(dev)
        ranks = sorted({r for q in qs for r in _ranks(n, q)})     # ranks np.quantile reads
        val = {r: np.float32(torch.kthvalue(x, r + 1).values.item()) for r in ranks}
        del x
        stand = np.empty(n, np.float32)                    # sorted stand-in array
        prev = 0
        for r in ranks:
            stand[prev:r + 1] = val[r]
            prev = r + 1
        stand[prev:] = val[ranks[-1]]
        # pass q as a list, like thresholds_all (NumPy 2 interpolates list vs scalar q
        # in different precisions)
        v = np.quantile(stand, list(qs))
        for i, q in enumerate(qs):
            out[q].append(v[i])
        del stand
    if tmode == "global":
        return {q: np.float32(out[q][0]) for q in qs}
    return {q: np.asarray(out[q]).astype(np.float32) for q in qs}


def _ranks(n, q):
    """Sorted positions np.quantile may read for quantile q of n values (floor/ceil +-1)."""
    v = q * (n - 1)
    lo = int(np.floor(v))
    return [max(lo - 1, 0), lo, min(lo + 1, n - 1), min(lo + 2, n - 1)]
