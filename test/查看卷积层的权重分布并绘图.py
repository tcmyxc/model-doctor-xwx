import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

import torch
import torchvision
import models
import loaders
import argparse
import os
import datetime
import time
import matplotlib
import yaml
import math


from torch import optim
from configs import config
from utils.lr_util import get_lr_scheduler
from utils.time_util import print_time, get_current_time
from sklearn.metrics import classification_report
from loss.refl import reduce_equalized_focal_loss
from loss.fl import focal_loss
from loss.hclv2 import hc_loss  # modify
from modify_kernel.util.draw_util import draw_lr
from modify_kernel.util.cfg_util import print_yml_cfg
from functools import partial
from utils.args_util import print_args
from utils.general import init_seeds

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

model = models.resnetv2.resnet32(num_classes=10)
weight_path = "/nfs/xwx/model-doctor-xwx/output/model/pretrained/resnet32-cifar-10-lt-ir100-refl-th-0.4-wr/checkpoint.pth"

model.load_state_dict(torch.load(weight_path)["model"])
conv_weight = model.layer3[4].conv2.weight.detach().numpy()
# print(conv_weight.shape) # out, in, k, k
conv_weight = np.sum(conv_weight * conv_weight, axis=(1, 2, 3))
conv_weight = conv_weight**0.5

plt.plot(range(len(conv_weight)), conv_weight, label='conv weight')
plt.ylabel("l2 weight")
# plt.xticks(range(len(conv_weight)))
plt.title("conv weight")
plt.xlabel("kernel")
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig("conv_weight.png")
plt.clf()
    