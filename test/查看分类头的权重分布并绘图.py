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
weight_path = "/nfs/xwx/model-doctor-xwx/output/model/three-stage/resnet32/cifar-10-lt-ir100/lr0.1/th0.5/custom_lr_scheduler/ce_loss/2022-05-13_20-19-52/best-model-20220513-203228-acc0.7509.pth"

model.load_state_dict(torch.load(weight_path)["model"])
fc_weight = model.linear.weight.detach().numpy()
fc_weight = np.sum(fc_weight * fc_weight, axis=1)
fc_weight = fc_weight**0.5

plt.plot(range(len(fc_weight)), fc_weight, 'r', label='fc _weight')
plt.ylabel("l2 weight")
plt.legend(loc="center right")

imb_factor = 0.01  # 0.01, 0.02, 0.1 => 100, 50, 10
cls_num = 10  # 10 or 100
x = [i for i in range(0, cls_num)]
y = []
cnt = 0
if cls_num == 10:
    # cifar-10
    img_cnt = 5000
else:
    # cifar-100
    img_cnt = 500
for i in range(cls_num):
    tmp = int(img_cnt * imb_factor ** (i / (cls_num - 1)))
    cnt += tmp
    y.append(tmp)

plt.twinx().plot(x, y, 'b', label='class samples')
plt.xticks(range(cls_num))

plt.title("fc weight & class samples")
plt.xlabel("class")
plt.ylabel("class samples")
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig("fc_weight.png")
plt.clf()
    