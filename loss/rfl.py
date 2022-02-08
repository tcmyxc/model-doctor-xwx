# -*- coding: utf-8 -*-
# Reduced Focal Loss: https://arxiv.org/pdf/1903.01347v2.pdf
# author: 徐文祥(tcmyxc)

import torch
import torch.nn.functional as F


def reduced_focal_loss(logits,
                       labels,
                       gamma=2,
                       threshold=0.5,
                       reduction="mean"):
    """Reduced Focal Loss: https://arxiv.org/pdf/1903.01347v2.pdf"""

    ce_loss = F.cross_entropy(logits, labels, reduction="none")
    log_pt = -ce_loss
    pt = torch.exp(log_pt)

    low_th_weight = torch.ones_like(pt)
    high_th_weight = (1 - pt) ** gamma / (threshold) ** gamma
    weights = torch.where(pt < threshold, low_th_weight, high_th_weight)

    rfl = weights * ce_loss

    if reduction == "sum":
        rfl = rfl.sum()
    elif reduction == "mean":
        rfl = rfl.mean()
    else:
        raise ValueError(f"reduction '{reduction}' is not valid")
    return rfl


if __name__ == "__main__":
    logits = torch.tensor([[0.3, 0.6, 0.9, 1], [0.6, 0.4, 0.9, 0.5]])
    labels = torch.tensor([1, 3])
    print(reduced_focal_loss(logits, labels, threshold=0.219))