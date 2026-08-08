"""Graph Deviation Network (Deng & Hooi, AAAI 2021), faithful compact implementation.
Parameters are grouped so the FL stage can share operators and keep embeddings local:
  - LOCAL (personalized):  self.embedding            (N x d, mission-specific)
  - SHARED (federated):    self.W, self.att, self.out (dimension-independent operators)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GDN(nn.Module):
    def __init__(self, n_nodes: int, window: int, embed_dim: int = 64, topk: int = None):
        super().__init__()
        self.N = n_nodes
        self.w = window
        self.d = embed_dim
        self.topk = topk if topk is not None else (n_nodes - 1)

        # (1) Sensor embeddings  v_i  -- LOCAL / personalized
        self.embedding = nn.Embedding(n_nodes, embed_dim)

        # (3) Shared operators -- FEDERATED
        self.W = nn.Linear(window, embed_dim)          # shared linear map of each node's window
        self.att = nn.Linear(4 * embed_dim, 1, bias=False)  # attention vector a over [g_i || g_j]
        self.leaky = nn.LeakyReLU(0.2)
        self.out = nn.Sequential(                      # PER-NODE shared head: v_i ∘ z_i (d) -> scalar
            nn.Linear(embed_dim, embed_dim),           # dimension-independent => federated
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def shared_state(self):
        """Weights that are federated (everything except the embedding table)."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("embedding.")}

    def load_shared_state(self, shared):
        self.load_state_dict(shared, strict=False)

    def get_adj(self):
        # (2) Graph structure learning: cosine similarity of embeddings -> top-k directed edges
        v = self.embedding.weight
        vn = F.normalize(v, dim=1)
        sim = vn @ vn.t()
        sim = sim - torch.eye(self.N, device=v.device) * 9e15   # exclude self before top-k
        idx = sim.topk(self.topk, dim=1).indices
        A = torch.zeros(self.N, self.N, device=v.device)
        A.scatter_(1, idx, 1.0)          # A[i, j] = 1  ->  j is a neighbor of i
        return A

    def forward(self, x):                # x: (B, N, w)
        B = x.size(0)
        v = self.embedding.weight        # (N, d)
        Wx = self.W(x)                   # (B, N, d)
        g = torch.cat([v.unsqueeze(0).expand(B, -1, -1), Wx], dim=2)   # (B, N, 2d)

        gi = g.unsqueeze(2).expand(B, self.N, self.N, 2 * self.d)
        gj = g.unsqueeze(1).expand(B, self.N, self.N, 2 * self.d)
        e = self.leaky(self.att(torch.cat([gi, gj], dim=3)).squeeze(-1))  # (B, N, N)

        A = self.get_adj() + torch.eye(self.N, device=x.device)          # add self-loop
        e = e.masked_fill(A.unsqueeze(0) == 0, float("-inf"))
        alpha = torch.softmax(e, dim=2)                                  # attention over neighbors j
        z = torch.relu(torch.bmm(alpha, Wx))                            # (B, N, d)

        vz = v.unsqueeze(0) * z                                          # element-wise v_i ∘ z_i  (B, N, d)
        return self.out(vz).squeeze(-1)                                 # per-node head -> (B, N) predictions
