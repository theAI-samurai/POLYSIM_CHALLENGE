"""
losses.py — Training objectives for multimodal speaker identification.

SupConLoss      — Supervised Contrastive Loss (Khosla et al., 2020)
                  Pulls same-speaker face/audio embeddings together in the
                  shared space, regardless of modality.

OrthogonalityLoss — Penalises cosine similarity between embeddings of
                    different speakers (FOP-style constraint).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    For each anchor, all samples sharing the same label are treated as
    positives. The loss maximises agreement between an anchor and its
    positives relative to all negatives in the batch.

    Args:
        temperature: Logit scale (lower = sharper distribution).

    Reference:
        Khosla et al. "Supervised Contrastive Learning." NeurIPS 2020.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D]  (need not be pre-normalised)
            labels:     [B]     long tensor of speaker indices
        Returns:
            Scalar loss.
        """
        B = embeddings.shape[0]
        emb = F.normalize(embeddings, dim=1)                      # [B, D]
        sim = torch.matmul(emb, emb.T) / self.temperature         # [B, B]

        # Masks
        eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)     # [B, B]
        pos_mask = label_eq & ~eye                                 # same class, not self

        # Numerical stability: subtract row max
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim)

        # Denominator: all pairs except self
        denom = (exp_sim * ~eye).sum(dim=1, keepdim=True) + 1e-8  # [B, 1]
        log_prob = sim - torch.log(denom)                          # [B, B]

        # Mean over positives per anchor
        pos_count = pos_mask.float().sum(dim=1).clamp(min=1)      # [B]
        loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count

        # Skip anchors that have no positive in the batch
        valid = pos_mask.any(dim=1)
        if valid.any():
            return loss[valid].mean()
        return loss.mean()


class OrthogonalityLoss(nn.Module):
    """Orthogonality constraint between different-speaker embeddings.

    Penalises the squared cosine similarity between embeddings of
    different speakers, encouraging them to be orthogonal in the
    shared embedding space.

    Reference:
        Nagrani et al. "Learnable Pins: Cross-Modal Embeddings for Person
        Identity." ECCV 2018.  (FOP variant)
    """

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D]
            labels:     [B]
        Returns:
            Scalar loss.
        """
        emb = F.normalize(embeddings, dim=1)                       # [B, D]
        sim = torch.matmul(emb, emb.T)                            # [B, B]

        diff_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()  # [B, B]
        loss = (sim ** 2 * diff_mask).sum() / (diff_mask.sum() + 1e-8)
        return loss
