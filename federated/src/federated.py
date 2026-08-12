"""Cross-mission federated GDN: Mission1 (6ch) + Mission2 (11ch).
Shared operators (W, attention, output MLP) are federated with FedProx + trimmed-mean;
per-sensor embeddings stay LOCAL (personalized) — so the two different channel counts
never need to be reconciled. Compares SOLO (each mission alone) vs FEDERATED."""
import os, sys, time, argparse, copy
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from gdn import GDN
from common import load_mission, WindowDS, score_on_test_official as score_on_test

torch.manual_seed(0); np.random.seed(0)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def make_loader(arr, w, train_windows, bs):
    stride = max(1, (len(arr) - w) // train_windows)
    return DataLoader(WindowDS(arr, w, stride=stride), batch_size=bs, shuffle=True)


def local_train(model, dl, epochs, lr, mu_prox=0.0, global_shared=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    gs = {k: v.to(DEVICE) for k, v in global_shared.items()} if (global_shared and mu_prox > 0) else None
    model.train()
    for _ in range(epochs):
        for x, y in dl:
            opt.zero_grad()
            loss = lossf(model(x.to(DEVICE)), y.to(DEVICE))
            if gs is not None:
                params = dict(model.named_parameters())
                prox = sum(((params[k] - v) ** 2).sum() for k, v in gs.items() if k in params)
                loss = loss + 0.5 * mu_prox * prox
            loss.backward(); opt.step()


def trimmed_mean(states, trim=0):
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
    ap.add_argument("--w", type=int, default=100)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--ft_epochs", type=int, default=4)        # per-mission personalization fine-tune
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mu", type=float, default=0.01)          # FedProx proximal strength
    ap.add_argument("--trim", type=int, default=0)             # trimmed-mean (0 => mean; needs >2 clients)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--train_windows", type=int, default=300000)
    ap.add_argument("--smooth", type=int, default=10)
    args = ap.parse_args()
    E_total = args.rounds * args.local_epochs + args.ft_epochs   # SOLO gets same total for fairness
    t0 = time.time()
    print(f"device={DEVICE} | fed: {args.rounds} rounds x {args.local_epochs} local + {args.ft_epochs} finetune "
          f"= {E_total} epochs (SOLO gets {E_total}) | FedProx mu={args.mu} trim={args.trim}", flush=True)

    # ---- load both missions ----
    M = {k: load_mission(k) for k in ["M1", "M2"]}
    for k in M:
        M[k]["dl"] = make_loader(M[k]["tr"], args.w, args.train_windows, args.bs)
        print(f"  {k}: N={M[k]['N']} channels, train rows={len(M[k]['tr'])}, "
              f"~{len(M[k]['dl'].dataset)} windows", flush=True)

    # ============ SOLO baselines ============
    print("\n=== SOLO (each mission trained independently) ===", flush=True)
    solo = {}
    for k in ["M1", "M2"]:
        torch.manual_seed(0)
        model = GDN(M[k]["N"], args.w, embed_dim=args.embed).to(DEVICE)
        local_train(model, M[k]["dl"], E_total, args.lr)
        res = score_on_test(model, M[k], args.w, DEVICE, smooth=args.smooth)
        solo[k] = res["EW_F_0.50"]
        print(f"  SOLO {k}: EW_F0.5={res['EW_F_0.50']:.4f} (P={res['EW_precision']:.3f} "
              f"R={res['EW_recall']:.3f}) | {time.time()-t0:.0f}s", flush=True)

    # ============ FEDERATED ============
    print("\n=== FEDERATED (cross-mission: shared operators, personalized embeddings) ===", flush=True)
    torch.manual_seed(0)
    models = {k: GDN(M[k]["N"], args.w, embed_dim=args.embed).to(DEVICE) for k in ["M1", "M2"]}
    global_shared = {k: v.cpu().clone() for k, v in models["M1"].shared_state().items()}
    for k in models:
        models[k].load_shared_state(global_shared)

    for r in range(args.rounds):
        updates = []
        for k in ["M1", "M2"]:
            models[k].load_shared_state(global_shared)
            local_train(models[k], M[k]["dl"], args.local_epochs, args.lr,
                        mu_prox=args.mu, global_shared=global_shared)
            updates.append({kk: v.detach().cpu().clone() for kk, v in models[k].shared_state().items()})
        global_shared = trimmed_mean(updates, trim=args.trim)
        print(f"  round {r+1}/{args.rounds} aggregated | {time.time()-t0:.0f}s", flush=True)

    # per-mission personalization: fine-tune the federated shared operators on local data
    fed = {}
    for k in ["M1", "M2"]:
        models[k].load_shared_state(global_shared)
        if args.ft_epochs > 0:
            local_train(models[k], M[k]["dl"], args.ft_epochs, args.lr)   # no prox, no aggregation
        res = score_on_test(models[k], M[k], args.w, DEVICE, smooth=args.smooth)
        fed[k] = res["EW_F_0.50"]
        print(f"  FED  {k}: EW_F0.5={res['EW_F_0.50']:.4f} (P={res['EW_precision']:.3f} "
              f"R={res['EW_recall']:.3f}) | {time.time()-t0:.0f}s", flush=True)

    # ============ comparison ============
    print("\n================ RESULT (EW corrected event-wise F0.5) ================", flush=True)
    print(f"{'mission':<8}{'SOLO':>10}{'FEDERATED':>12}{'delta':>10}", flush=True)
    for k in ["M1", "M2"]:
        d = fed[k] - solo[k]
        print(f"{k:<8}{solo[k]:>10.4f}{fed[k]:>12.4f}{d:>+10.4f}", flush=True)
    print(f"\nDONE total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
