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

res32 = models.resnetv2.resnet32()

for i, param in enumerate(res32.parameters()):
        if i < 93:
                param.requires_grad = False
parameters = [p for p in res32.parameters() if p.requires_grad]
print(f"\n[DEBUG] requires_grad parameters: {parameters}")