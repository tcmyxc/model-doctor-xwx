import torch
import torch.nn.functional as F
import numpy as np


def channel_loss(logits, labels, channel_mask, reduction="mean"):
    channel_mask = 1- torch.from_numpy(np.asarray(channel_mask))
    
