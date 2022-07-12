import torch
import torch.nn.functional as F
import numpy as np


def balanced_softmax_loss(logits, labels, sample_per_class, reduction="mean"):
    sample_per_class = torch.from_numpy(np.asarray(sample_per_class))
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    bsl = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return bsl