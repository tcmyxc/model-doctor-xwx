# -*- coding: utf-8 -*-

def loss_reduction(loss, reduction="mean"):
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"reduction '{reduction}' is not valid")