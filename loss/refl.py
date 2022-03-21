# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def reduce_equalized_focal_loss(logits,
                                targets,
                                gamma=2,
                                scale_factor=8,
                                threshold=0.5,
                                reduction="mean"):
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    outputs = F.cross_entropy(logits, targets)  # 求导使用，不能带 reduction 参数
    log_pt = -ce_loss
    pt = torch.exp(log_pt)

    targets = targets.view(-1, 1)  # 多加一个维度，为使用 gather 函数做准备
    grad_i = torch.autograd.grad(outputs=-outputs, inputs=logits)[0]  # 求导
    grad_i = grad_i.gather(1, targets)  # 每个类对应的梯度
    pos_grad_i = F.relu(grad_i).sum()
    neg_grad_i = F.relu(-grad_i).sum()
    neg_grad_i += 1e-9  # 防止除数为0
    grad_i = pos_grad_i / neg_grad_i
    grad_i = torch.clamp(grad_i, min=0, max=1)  # 裁剪梯度

    dy_gamma = gamma + scale_factor * (1 - grad_i)
    dy_gamma = dy_gamma.view(-1)  # 去掉多的一个维度
    # weighting factor
    wf = dy_gamma / gamma

    low_th_weight = torch.ones_like(pt)
    high_th_weight = (1 - pt) ** gamma / (threshold) ** gamma
    weights = torch.where(pt < threshold, low_th_weight, high_th_weight)

    rfl = wf * weights * ce_loss

    if reduction == "sum":
        rfl = rfl.sum()
    elif reduction == "mean":
        rfl = rfl.mean()
    else:
        raise ValueError(f"reduction '{reduction}' is not valid")
    return rfl


if __name__ == "__main__":
    logits = torch.tensor([[0.3, 0.6, 0.9, 1], [0.6, 0.4, 0.9, 0.5]], requires_grad=True)
    targets = torch.tensor([1, 3])
    print(reduce_equalized_focal_loss(logits, targets, threshold=0.5))
