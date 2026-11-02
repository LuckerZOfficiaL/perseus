#!/usr/bin/env python
"""late-inject-attention: the hand-over as a cross-attention. The placeholder tokens' states (queries) read the secret
tokens' raw embeddings (keys and values); the output is added to the placeholder states. One block per hand-over
depth k_late. Identity initialisation (q, k, v, o all identity): at step 0 each placeholder's query is its own state,
whose position embedding matches its own secret's, so the block starts close to late-inject-additive."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFusion(nn.Module):
    def __init__(self, D, H, klates):
        super().__init__()
        self.D, self.H, self.klates = D, H, [int(k) for k in klates]
        self.blocks = nn.ModuleDict()
        for k in self.klates:
            blk = nn.ModuleDict({n: nn.Linear(D, D) for n in ("q", "k", "v", "o")})
            for lin in blk.values():
                nn.init.eye_(lin.weight)
                nn.init.zeros_(lin.bias)
            self.blocks[str(k)] = blk

    def _split(self, x):
        B, n, D = x.shape
        return x.view(B, n, self.H, D // self.H).transpose(1, 2)

    def forward(self, k_late, h, e, key_mask=None):
        """h: (B, n, D) placeholder states (queries); e: (B, n, D) raw embeddings of the secret tokens (keys, values);
        key_mask: (B, n) bool, which key entries are real. Returns the update to add to h."""
        blk = self.blocks[str(int(k_late))]
        q, k, v = self._split(blk["q"](h)), self._split(blk["k"](e)), self._split(blk["v"](e))
        mask = None if key_mask is None else key_mask[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        B, H, n, dh = a.shape
        return blk["o"](a.transpose(1, 2).reshape(B, n, H * dh))

    def save(self, path):
        torch.save(dict(D=self.D, H=self.H, klates=self.klates, state=self.state_dict()), path)

    @classmethod
    def load(cls, path, device):
        d = torch.load(path, map_location=device)
        m = cls(d["D"], d["H"], d["klates"]).to(device)
        m.load_state_dict(d["state"])
        return m.eval()
