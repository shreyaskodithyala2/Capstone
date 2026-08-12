"""GDNv2: graph-attention forecaster with a long-context temporal encoder.

  - Temporal encoder: causal dilated TCN, receptive field 253 steps (~2.1 h on M1's
    30 s grid). Convolutional, so inference runs over the FULL series in one pass --
    every position's encoding is computed once (a per-window GRU would recompute a
    250-step recurrence at each of 7.36M positions).
  - Availability features (gap_flag, log time-since-last-sample) as input channels.
  - Telecommand nodes: input-only exogenous nodes; channels attend to them, never
    the reverse (a sensor cannot cause a command).
  - Federation-ready split: `embedding` local; everything else dimension-independent.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RF = 253   # receptive field: 1 + (k-1)*sum(dilations) = 1 + 4*63


class TCN(nn.Module):
    def __init__(self, feat=3, ch=32, d=64, k=5):
        super().__init__()
        dil = [1, 2, 4, 8, 16, 32]
        self.pads = [(k - 1) * dd for dd in dil]
        self.convs = nn.ModuleList(
            [nn.Conv1d(feat if i == 0 else ch, ch, k, dilation=dd) for i, dd in enumerate(dil)])
        self.proj = nn.Conv1d(ch, d, 1)

    def forward(self, x):                      # x: (B, F, T) -> (B, d, T), causal
        h = x
        for conv, p in zip(self.convs, self.pads):
            inp = F.pad(h, (p, 0))
            out = conv(inp)
            h = F.relu(out) + (h if h.shape == out.shape else 0)
        return self.proj(h)


class GDNv2(nn.Module):
    def __init__(self, n_ch, n_tc, feat=3, embed_dim=64, conv_ch=32, topk=None):
        super().__init__()
        self.n_ch, self.n_tc = n_ch, n_tc
        self.N = n_ch + n_tc
        self.d = embed_dim
        self.topk = topk if topk is not None else max(n_ch - 1, 1)
        self.embedding = nn.Embedding(self.N, embed_dim)           # LOCAL
        self.tcn = TCN(feat, conv_ch, embed_dim)                   # SHARED
        # decomposed GAT scoring: e_ij = leaky(a1.g_i + a2.g_j)  (memory-light)
        self.a1 = nn.Linear(2 * embed_dim, 1, bias=False)
        self.a2 = nn.Linear(2 * embed_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.head = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.ReLU(),
                                  nn.Linear(embed_dim, 1))

    def shared_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("embedding.")}

    def load_shared_state(self, s):
        self.load_state_dict(s, strict=False)

    def get_adj(self):
        v = self.embedding.weight
        A = torch.zeros(self.N, self.N, device=v.device)
        vc = F.normalize(v[:self.n_ch], dim=1)
        sim = vc @ vc.t() - torch.eye(self.n_ch, device=v.device) * 9e15
        k = min(self.topk, self.n_ch - 1)
        A[:self.n_ch, :self.n_ch].scatter_(1, sim.topk(k, 1).indices, 1.0)
        if self.n_tc:
            A[:self.n_ch, self.n_ch:] = 1.0
        A += torch.eye(self.N, device=v.device)
        return (A > 0).float()

    def encode(self, x):                       # x: (B, N, T, F) -> h: (B, N, T, d)
        B, N, T, Fd = x.shape
        h = self.tcn(x.reshape(B * N, T, Fd).transpose(1, 2))
        return h.transpose(1, 2).reshape(B, N, T, self.d)

    def predict_positions(self, h_pos):        # h_pos: (P, N, d) -> (P, n_ch)
        P = h_pos.size(0)
        v = self.embedding.weight.unsqueeze(0).expand(P, -1, -1)
        g = torch.cat([v, h_pos], dim=2)                       # (P, N, 2d)
        e = self.leaky(self.a1(g) + self.a2(g).transpose(1, 2))  # (P, N, N)
        A = self.get_adj()
        e = e.masked_fill(A.unsqueeze(0) == 0, float("-inf"))
        alpha = torch.softmax(e, dim=2)
        z = torch.relu(torch.bmm(alpha, h_pos))                # (P, N, d)
        gz = torch.cat([self.embedding.weight.unsqueeze(0).expand(P, -1, -1) * z, z], dim=2)
        return self.head(gz).squeeze(-1)[:, :self.n_ch]

    def forward_segments(self, x):
        """x: (B, N, L, F) training segments. Returns preds for positions RF-1..L-2
        (predicting t+1), shape (B, L-RF, n_ch)."""
        B, N, L, Fd = x.shape
        h = self.encode(x)                                     # (B, N, L, d)
        hp = h[:, :, RF - 1:L - 1]                             # (B, N, L', d)
        Lp = hp.size(2)
        hp = hp.permute(0, 2, 1, 3).reshape(B * Lp, N, self.d)
        return self.predict_positions(hp).reshape(B, Lp, self.n_ch)


@torch.no_grad()
def series_errors(model, X, device, enc_chunk=200_000, att_chunk=16_384):
    """Full-series forecast errors. X: (T, N, F) numpy. Returns (T, n_ch) float32
    with zeros in the first RF rows (no full context there)."""
    model.eval()
    T, N, Fd = X.shape
    n_ch = model.n_ch
    errs = np.zeros((T, n_ch), np.float32)
    for a in range(RF, T, enc_chunk):
        b = min(a + enc_chunk, T)                          # predict target rows a..b-1
        lo = a - RF                                        # h[t-1] predicts row t
        seg = torch.from_numpy(np.ascontiguousarray(
            np.transpose(X[lo:b], (1, 0, 2))[None])).to(device)   # (1, N, T', F)
        h = model.encode(seg)[0]                            # (N, T', d)
        hp = h[:, RF - 1:RF - 1 + (b - a)].permute(1, 0, 2)  # global pos a-1..b-2 -> (P, N, d)
        P = hp.size(0)
        preds = torch.empty(P, n_ch, device=device)
        for s in range(0, P, att_chunk):
            e = min(s + att_chunk, P)
            preds[s:e] = model.predict_positions(hp[s:e])
        tgt = torch.from_numpy(X[a:b, :n_ch, 0]).to(device)
        errs[a:b] = (preds[:b - a] - tgt).abs().cpu().numpy()
        del seg, h, hp, preds
    return errs
