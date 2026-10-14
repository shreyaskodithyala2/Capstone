"""The earlier forecasting model used by the federated prototype: a compact Graph Deviation
Network (Deng & Hooi, AAAI 2021) that reads a fixed window of past values per channel.

    input  x (B, N, w)  last w normalised values of each of N channels
    output   (B, N)     predicted next value of each channel

Built for federation: the embedding table (N x d) depends on the mission's channel count and
stays LOCAL; the window map W, the attention vector and the per-node output head have the same
shape for any mission and are SHARED. (The current model, src/model.py, keeps the same split.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowGDN(nn.Module):
    def __init__(self, n_nodes, window, embed_dim=64, topk=None):
        super().__init__()
        self.N, self.w, self.d = n_nodes, window, embed_dim
        self.topk = topk if topk is not None else (n_nodes - 1)
        self.embedding = nn.Embedding(n_nodes, embed_dim)          # LOCAL: one vector per channel
        self.W = nn.Linear(window, embed_dim)                      # SHARED: window -> d
        self.att = nn.Linear(4 * embed_dim, 1, bias=False)         # SHARED: score over [g_i, g_j]
        self.leaky = nn.LeakyReLU(0.2)
        self.out = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                 nn.Linear(embed_dim, 1))          # SHARED: same head per node

    def shared_state(self):
        """Weights sent to the server: everything except the embedding table."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("embedding.")}

    def load_shared_state(self, shared):
        """Load aggregated shared weights; the local embedding is kept."""
        self.load_state_dict(shared, strict=False)

    def get_adj(self):
        """(N, N) 0/1 graph: each channel links to its top-k most similar channels (cosine
        similarity of the learned embeddings)."""
        v = self.embedding.weight
        vn = F.normalize(v, dim=1)
        sim = vn @ vn.t() - torch.eye(self.N, device=v.device) * 9e15   # no self-match
        A = torch.zeros(self.N, self.N, device=v.device)
        A.scatter_(1, sim.topk(self.topk, dim=1).indices, 1.0)          # A[i, j] = 1: j feeds i
        return A

    def forward(self, x):
        B = x.size(0)
        v = self.embedding.weight                                       # (N, d)
        Wx = self.W(x)                                                  # (B, N, d)
        g = torch.cat([v.unsqueeze(0).expand(B, -1, -1), Wx], dim=2)   # (B, N, 2d)
        gi = g.unsqueeze(2).expand(B, self.N, self.N, 2 * self.d)
        gj = g.unsqueeze(1).expand(B, self.N, self.N, 2 * self.d)
        e = self.leaky(self.att(torch.cat([gi, gj], dim=3)).squeeze(-1))   # (B, N, N) scores
        A = self.get_adj() + torch.eye(self.N, device=x.device)          # + self loop
        e = e.masked_fill(A.unsqueeze(0) == 0, float("-inf"))
        alpha = torch.softmax(e, dim=2)                                  # weights over neighbours
        z = torch.relu(torch.bmm(alpha, Wx))                             # (B, N, d) context
        return self.out(v.unsqueeze(0) * z).squeeze(-1)                 # (B, N) predictions
