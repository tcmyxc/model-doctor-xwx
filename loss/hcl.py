# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def hc_loss(logits, labels, reduction="mean"):
    ce_loss = F.cross_entropy(logits, labels, reduction="none")
    incorrect_inx = (logits.argmax(1) != labels)
    hc_loss = ce_loss[incorrect_inx]

    if min(hc_loss.shape) == 0:
        hc_loss = torch.zeros_like(ce_loss)

    if reduction == "sum":
        hc_loss = hc_loss.sum()
    elif reduction == "mean":
        hc_loss = hc_loss.mean()
    else:
        raise ValueError(f"reduction '{reduction}' is not valid")
    return hc_loss

    


if __name__ == "__main__":
    logits = torch.tensor([[0.3, 0.6, 0.9, 1], [0.6, 0.4, 0.9, 0.5]])
    labels = torch.tensor([1, 2])
    print(hc_loss(logits, labels))