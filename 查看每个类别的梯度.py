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
from modify_kernel.util.draw_util import draw_lr
from modify_kernel.util.cfg_util import print_yml_cfg
from functools import partial
from utils.args_util import print_args

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CFG:
    data_name = "cifar-10-lt-ir100"
    in_channels = 3
    num_classes = 10

    model_name = "resnet32"
    model_path = "/nfs/xwx/model-doctor-xwx/output/model/pretrained/resnet32-cifar-10-lt-ir100-refl-th-0.4-wr/checkpoint.pth"
    

def cal_cls_grad(logits, targets, num_classes):
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    outputs = F.cross_entropy(logits, targets)  # 求导使用，不能带 reduction 参数
    log_pt = -ce_loss
    pt = torch.exp(log_pt)

    targets = targets.view(-1, 1)  # 多加一个维度，为使用 gather 函数做准备
    grad_i = torch.autograd.grad(outputs=-outputs, inputs=logits)[0]  # 求导
    # print(grad_i)
    grad_i = grad_i.gather(1, targets)  # 每个类对应的梯度
    # print(grad_i)
    # print(grad_i / grad_i.sum())

    # weights = 1 - torch.tanh(grad_i)
    # return (weights * ce_loss).mean()

    grads = [grad_i[targets == i].sum().detach().cpu().item() for i in range(num_classes)]
    grads = torch.tensor(grads)
    grads_percent = grads / grads.sum()
    # print(grads_percent)
    tanh_grads_percent = 1 - torch.tanh(grads_percent)
    grads_weight = tanh_grads_percent
    # print(tanh_grads_percent)
    return grads_weight



 # device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\n[INFO] train on ', device)

cfg = CFG()

loss_fn = nn.CrossEntropyLoss()

data_loaders, _ = loaders.load_data(data_name=cfg.data_name)
model = models.load_model(
        model_name=cfg.model_name,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes
)
model.load_state_dict(torch.load(cfg.model_path)["model"])
model.to(device)
optimizer = optim.SGD(
    params=model.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4,
)

for batch, (X, y, _) in enumerate(data_loaders["train"]):
    X, y = X.to(device), y.to(device)
    with torch.set_grad_enabled(True):
        # Compute prediction error
        pred, feature_out = model(X)  # 网络前向计算
        grads_weight = cal_cls_grad(pred, y, cfg.num_classes)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        for name, parms in model.named_parameters():
            if len(parms.grad.size()) == 4:
                print(parms.grad.size())
        optimizer.step()  # 更新参数
    break




