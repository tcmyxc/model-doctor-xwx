# -*- coding: utf-8 -*-
# Adaptive Class Weight based Dual Focal Loss for Improved Semantic Segmentation

import torch
import torch.nn.functional as F

# import sys
# sys.path.append('/mnt/hangzhou_116_homes/xwx/model-doctor-xwx')

from loss.loss_util import loss_reduction


def dual_focal_loss(logits, labels, reduction="mean"):
    r""" 
    Adaptive Class Weight based Dual Focal Loss for Improved Semantic
    Segmentation 
    """
    ce_loss = F.cross_entropy(logits, labels, reduction="none")
    log_pt = -ce_loss
    pt = torch.exp(log_pt)
    dfl = ce_loss + (1-pt)

    return loss_reduction(dfl, reduction)


if __name__ == "__main__":
    logits = torch.tensor([[0.3, 0.6, 0.9, 1], [0.6, 0.4, 0.9, 0.5]])
    labels = torch.tensor([1, 3])
    print(dual_focal_loss(logits, labels))

