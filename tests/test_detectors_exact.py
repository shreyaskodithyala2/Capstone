"""Proves the fast (vectorised) functions in src/detectors.py give EXACTLY the same output as
the original slow per-channel versions that produced the published results.

Each check compares with np.array_equal (bit-for-bit, not "close enough") on synthetic
errors with quiet stretches and spikes. Runs in seconds:

    python tests/test_detectors_exact.py        # prints "ALL EXACT" on success
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from detectors import (block_stats, block_z, multi_span_z, ewma,
                       close_runs, open_runs, gap_pred_ch,
                       close_runs2d, open_runs2d, gap_pred_ch2d)

rng = np.random.default_rng(0)


# ---- reference implementations: the original one-channel-at-a-time code ----

def ref_block_stats(e, block):
    """Original 1-D block_stats."""
    T = len(e); nb = max(int(np.ceil(T / block)), 2)
    idx = np.minimum(np.arange(nb) * block + block // 2, T - 1)
    med = np.empty(nb, np.float32); iqr = np.empty(nb, np.float32)
    for i in range(nb):
        seg = e[i * block: min((i + 1) * block, T)]
        if len(seg) == 0:
            med[i], iqr[i] = 0.0, 1.0; continue
        q1, q2, q3 = np.percentile(seg, [25, 50, 75])
        med[i], iqr[i] = q2, max(q3 - q1, 1e-6)
    x = np.arange(T)
    return np.interp(x, idx, med).astype(np.float32), np.interp(x, idx, iqr).astype(np.float32)


def ref_block_z(errs, block, floor_q=0.25):
    """Original block_z, looping over channels."""
    T, C = errs.shape; Z = np.empty((T, C), np.float32)
    for c in range(C):
        med, iqr = ref_block_stats(errs[:, c], block)
        gf = max(float(np.quantile(iqr, floor_q)), 1e-6)
        Z[:, c] = (errs[:, c] - med) / np.maximum(iqr, gf)
    return Z


def ref_multi_span_z(errs, spans, block, floor_q=0.25):
    """Original multi_span_z, no channel chunking."""
    out = None
    for sp in spans:
        z = ref_block_z(ewma(errs, sp), block, floor_q)
        out = z if out is None else np.maximum(out, z)
    return out


def main():
    T, C, block = 9000, 9, 700
    # heavy-tailed noise with a long shift and a short spike: the case block_z is built for
    E = np.abs(rng.standard_normal((T, C))).astype(np.float32) * 0.01
    E[3000:3100] += 5.0
    E[6000:6003] += 50.0

    # Detector B pieces
    for name, got, exp in [
        ("block_stats(1d).med", block_stats(E[:, 0], block)[0], ref_block_stats(E[:, 0], block)[0]),
        ("block_stats(1d).iqr", block_stats(E[:, 0], block)[1], ref_block_stats(E[:, 0], block)[1]),
        ("block_z",             block_z(E, block),              ref_block_z(E, block)),
    ]:
        assert np.array_equal(got, exp), (name, np.abs(got - exp).max())
        print(f"  OK  {name:24s} exact")

    for ch in (1, 4, 9, 16, 32):                # result must not depend on the chunk size
        g = multi_span_z(E, [1, 3, 15, 60, 240], block, chunk=ch)
        assert np.array_equal(g, ref_multi_span_z(E, [1, 3, 15, 60, 240], block))
        print(f"  OK  multi_span_z chunk={ch:<3d}       exact")

    # cleanup steps, including runs that touch the first and last row
    P = (rng.random((T, C)) < 0.02).astype(np.uint8)
    P[0:5] = 1; P[-5:] = 1
    P[1000:1400] = 1
    for gap in (0, 1, 2, 3, 60, 240):
        exp = np.stack([close_runs(P[:, c].copy(), gap) for c in range(C)], 1)
        assert np.array_equal(close_runs2d(P, gap), exp), f"close gap={gap}"
        print(f"  OK  close_runs2d max_gap={gap:<4d}   exact")
    for ml in (1, 2, 3, 5, 10, 400):
        exp = np.stack([open_runs(P[:, c].copy(), ml) for c in range(C)], 1)
        assert np.array_equal(open_runs2d(P, ml), exp), f"open ml={ml}"
        print(f"  OK  open_runs2d  min_len={ml:<4d}   exact")

    # Detector C
    cnt = (rng.random((T, C)) < 0.3).astype(np.uint16)
    cnt[2000:2050] = 0; cnt[:20] = 0; cnt[-30:] = 0
    for L in (1, 2, 10, 40):
        assert np.array_equal(gap_pred_ch2d(cnt, L), gap_pred_ch(cnt, L)), f"gap L={L}"
        print(f"  OK  gap_pred_ch2d L={L:<8d}     exact")

    # the order used by assemble(): open(close(x))
    for gap, ml in [(60, 3), (240, 1), (0, 3)]:
        exp = np.stack([open_runs(close_runs(P[:, c].copy(), gap), ml) for c in range(C)], 1)
        assert np.array_equal(open_runs2d(close_runs2d(P, gap), ml), exp)
        print(f"  OK  open(close) gap={gap} ml={ml}      exact")
    print("\nALL EXACT")


if __name__ == "__main__":
    main()
