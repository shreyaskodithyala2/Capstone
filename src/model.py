"""The forecasting model: a causal TCN encoder followed by a Graph Deviation Network (GDN).

What it does, per timestep t:
  1. TCN       : reads the last 253 steps of each node's 3 features -> a 64-d vector h.
  2. Graph     : each channel attends to its top-k most similar channels, every telecommand
                 node and itself, and mixes their vectors into a context vector z.
  3. Head      : a small MLP turns [embedding * z, z] into one number = predicted value at t+1.
The anomaly signal is the forecast error |predicted - actual| (see series_errors).

Federated-learning split: `embedding` is mission-specific (its size depends on the number
of nodes); every other weight has the same shape for any mission and can be shared.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RF = 253   # receptive field in steps: 1 + (kernel-1) * sum(dilations) = 1 + 4*63 (~2.1 h on M1)


class TCN(nn.Module):
    """Causal dilated temporal convolution network, shared by every node.

    6 layers, kernel 5, dilations 1-2-4-8-16-32, 32 channels, then a 1x1 conv to 64.
    Input : (B, 3, T)  one node's feature series per batch row
    Output: (B, 64, T) where output[t] depends only on inputs t-252 .. t (never the future)
    """
    def __init__(self, feat=3, ch=32, d=64, k=5):
        super().__init__()
        dil = [1, 2, 4, 8, 16, 32]
        self.pads = [(k - 1) * dd for dd in dil]          # left padding that keeps it causal
        self.convs = nn.ModuleList(
            [nn.Conv1d(feat if i == 0 else ch, ch, k, dilation=dd) for i, dd in enumerate(dil)])
        self.proj = nn.Conv1d(ch, d, 1)                   # 32 -> 64, applied once at the end

    def forward(self, x):
        h = x
        for conv, p in zip(self.convs, self.pads):
            inp = F.pad(h, (p, 0))                        # pad the PAST only -> no look-ahead
            out = conv(inp)
            # ReLU, then a residual skip when shapes match (all layers except the first, 3->32)
            h = F.relu(out) + (h if h.shape == out.shape else 0)
        return self.proj(h)


class GDNForecaster(nn.Module):
    """TCN encoder + learned sparse graph + attention + per-node MLP head.

    n_ch channel nodes are forecast; n_tc telecommand nodes are input-only (channels attend
    to them, never the other way round). Parameters for M1 lightweight: 38,049 in total, of
    which 36,961 are shareable across missions and 1,088 (the embedding table) are local.
    """
    def __init__(self, n_ch, n_tc, feat=3, embed_dim=64, conv_ch=32, topk=None):
        super().__init__()
        self.n_ch, self.n_tc = n_ch, n_tc
        self.N = n_ch + n_tc
        self.d = embed_dim
        # Each channel keeps its k most similar channels as neighbours. k = min(n_ch-1, 10):
        # 5 for M1 lightweight, 10 for M2 and the full sets. Capping at 10 keeps the graph
        # sparse at 76/100 channels (~one channel group plus margin) instead of complete.
        self.topk = topk if topk is not None else max(min(n_ch - 1, 10), 1)
        self.embedding = nn.Embedding(self.N, embed_dim)           # LOCAL: one vector per node
        self.tcn = TCN(feat, conv_ch, embed_dim)                   # SHARED
        # Attention score e_ij = LeakyReLU(a1.g_i + a2.g_j). Split into two vectors so the
        # (P, N, N, 128) pair tensor is never built -- only (P, N, N) scores.
        self.a1 = nn.Linear(2 * embed_dim, 1, bias=False)
        self.a2 = nn.Linear(2 * embed_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.head = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.ReLU(),
                                  nn.Linear(embed_dim, 1))         # same MLP for every node
        self._adj = None        # cached adjacency for eval, see get_adj()

    def shared_state(self):
        """Weights to send to the federated server: everything except the embedding table."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("embedding.")}

    def load_shared_state(self, s):
        """Load aggregated shared weights from the server; the local embedding is kept."""
        self.load_state_dict(s, strict=False)

    def train(self, mode=True):
        self._adj = None                     # embeddings will change -> drop the cached graph
        return super().train(mode)

    def get_adj(self):
        """Build the (N, N) 0/1 adjacency mask; row i = the nodes channel i may attend to.

        channel -> top-k channels by cosine similarity of their embeddings
        channel -> every telecommand node
        every node -> itself
        telecommand rows stay empty apart from the self loop (commands are never predicted)
        In eval mode the embeddings are frozen, so the mask is computed once and cached.
        """
        if not self.training and self._adj is not None:
            return self._adj
        v = self.embedding.weight
        A = torch.zeros(self.N, self.N, device=v.device)
        vc = F.normalize(v[:self.n_ch], dim=1)
        sim = vc @ vc.t() - torch.eye(self.n_ch, device=v.device) * 9e15   # cosine, no self
        k = min(self.topk, self.n_ch - 1)
        A[:self.n_ch, :self.n_ch].scatter_(1, sim.topk(k, 1).indices, 1.0)
        if self.n_tc:
            A[:self.n_ch, self.n_ch:] = 1.0
        A += torch.eye(self.N, device=v.device)
        A = (A > 0).float()
        if not self.training:
            self._adj = A
        return A

    def encode(self, x):
        """Run the TCN over every node. x (B, N, T, 3) -> h (B, N, T, 64).
        All B*N node series go through the same TCN weights."""
        B, N, T, Fd = x.shape
        h = self.tcn(x.reshape(B * N, T, Fd).transpose(1, 2))
        return h.transpose(1, 2).reshape(B, N, T, self.d)

    def predict_positions(self, h_pos):
        """Graph attention + head for a batch of P timesteps.

        Input : h_pos (P, N, 64) TCN vectors of every node at P positions
        Output: (P, n_ch) predicted next value (normalised) of each channel
        """
        P = h_pos.size(0)
        v = self.embedding.weight.unsqueeze(0).expand(P, -1, -1)
        g = torch.cat([v, h_pos], dim=2)                       # (P, N, 128) identity + state
        e = self.leaky(self.a1(g) + self.a2(g).transpose(1, 2))  # (P, N, N) raw scores
        A = self.get_adj()
        e = e.masked_fill(A.unsqueeze(0) == 0, float("-inf"))  # non-neighbours get weight 0
        alpha = torch.softmax(e, dim=2)                        # each row sums to 1
        z = torch.relu(torch.bmm(alpha, h_pos))                # (P, N, 64) weighted mix
        gz = torch.cat([self.embedding.weight.unsqueeze(0).expand(P, -1, -1) * z, z], dim=2)
        return self.head(gz).squeeze(-1)[:, :self.n_ch]        # drop telecommand outputs

    def forward_segments(self, x):
        """Training forward pass on segments.

        Input : x (B, N, L, 3) segments of L steps (L = 512 in training)
        Output: (B, L-RF, n_ch) predictions for steps RF..L-1, each made from the TCN
                vector one step earlier (so it is a next-step forecast)
        """
        B, N, L, Fd = x.shape
        h = self.encode(x)                                     # (B, N, L, d)
        hp = h[:, :, RF - 1:L - 1]                             # first position with full history
        Lp = hp.size(2)
        hp = hp.permute(0, 2, 1, 3).reshape(B * Lp, N, self.d)
        return self.predict_positions(hp).reshape(B, Lp, self.n_ch)


@torch.no_grad()
def series_errors(model, X, device, enc_chunk=200_000, att_chunk=None):
    """Forecast error of every channel at every timestep of a whole split.

    Input : X (T, N, 3) numpy, from dataset.build_arrays
    Output: (T, n_ch) float32 = |predicted value_z - actual value_z|; rows < RF are 0
    Example: M1 lightweight, X (7_364_161, 17, 3) -> (7_364_161, 6); train + test passes
             together take ~1-1.5 min on Apple MPS.

    Because the encoder is convolutional, the series is encoded in large chunks and every
    position is computed once. Attention memory grows with N^2, so att_chunk is sized to a
    fixed element budget (equal to the old fixed 16,384 positions at N = 17).
    """
    model.eval()
    T, N, Fd = X.shape
    if att_chunk is None:
        g = 8 if str(device).startswith("cuda") else 1        # CUDA cards hold 8x the budget
        att_chunk = int(np.clip(g * 16_384 * (17 ** 2) / (N * N), 512, g * 16_384))
    n_ch = model.n_ch
    errs = np.zeros((T, n_ch), np.float32)
    enc_chunk = max(int(enc_chunk * 17 / N), 20_000)   # encoder memory also scales with N
    for a in range(RF, T, enc_chunk):
        b = min(a + enc_chunk, T)                          # predict target rows a..b-1
        lo = a - RF                                        # include RF steps of history
        seg = torch.from_numpy(np.ascontiguousarray(
            np.transpose(X[lo:b], (1, 0, 2))[None])).to(device)   # (1, N, T', F)
        h = model.encode(seg)[0]                            # (N, T', d)
        hp = h[:, RF - 1:RF - 1 + (b - a)].permute(1, 0, 2)  # h[t-1] predicts row t -> (P, N, d)
        P = hp.size(0)
        preds = torch.empty(P, n_ch, device=device)
        for s in range(0, P, att_chunk):
            e = min(s + att_chunk, P)
            preds[s:e] = model.predict_positions(hp[s:e])
        tgt = torch.from_numpy(X[a:b, :n_ch, 0]).to(device)     # actual value_z
        errs[a:b] = (preds[:b - a] - tgt).abs().cpu().numpy()
        del seg, h, hp, preds
    return errs
