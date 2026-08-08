"""Train solo GDN per mission, score with Telemanom-pruning (official metric).
Compares val-tuned-threshold scoring vs Telemanom-pruning scoring."""
import os, sys, time, argparse
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from gdn import GDN
from common import load_mission, WindowDS, score_on_test_official, score_on_test_telemanom

torch.manual_seed(0); np.random.seed(0)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=100)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--train_windows", type=int, default=300000)
    ap.add_argument("--smooth", type=int, default=10)
    ap.add_argument("--p", type=float, default=0.13)
    args = ap.parse_args()
    t0 = time.time()
    lossf = torch.nn.MSELoss()

    for k in ["M1", "M2"]:
        M = load_mission(k)
        stride = max(1, (len(M["tr"]) - args.w) // args.train_windows)
        dl = DataLoader(WindowDS(M["tr"], args.w, stride=stride), batch_size=args.bs, shuffle=True)
        torch.manual_seed(0)
        model = GDN(M["N"], args.w, embed_dim=args.embed).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.train()
        for ep in range(args.epochs):
            for x, y in dl:
                opt.zero_grad(); loss = lossf(model(x.to(DEVICE)), y.to(DEVICE)); loss.backward(); opt.step()
        rt = score_on_test_telemanom(model, M, args.w, DEVICE, smooth=args.smooth, p=args.p)
        rv = score_on_test_official(model, M, args.w, DEVICE, smooth=args.smooth)
        print(f"{k}: TELEMANOM-PRUNE EW_F0.5={rt['EW_F_0.50']:.4f} (P={rt['EW_precision']:.3f} "
              f"R={rt['EW_recall']:.3f}, pos={rt['_positives']})  |  VAL-THRESH EW_F0.5={rv['EW_F_0.50']:.4f} "
              f"(P={rv['EW_precision']:.3f} R={rv['EW_recall']:.3f}) | {time.time()-t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
