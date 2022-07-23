"""
1、找头部类别中等响应的卷积核
2、求步骤1找到的卷积核和尾部类卷积核的交集
3、增大这些卷积核在尾部类别的梯度
"""

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
from loss.hcl import hc_loss
from modify_kernel.util.draw_util import draw_lr, draw_fc_weight
from modify_kernel.util.cfg_util import print_yml_cfg
from functools import partial
from utils.args_util import print_args
from utils.general import init_seeds, get_head_and_kernel, get_head_ratio

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import warnings # ignore warnings
warnings.filterwarnings("ignore")


np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "%.3g" % x))

class HookModule:
    """hook，对中间层的输入输出进行记录"""

    def __init__(self, model, module):
        """给model的某一层加钩子"""
        self.model = model
        self.activations = None

        module.register_forward_hook(self._hook_activations)

    # 钩子函数原型：hook(module, input, output) -> None or modified output
    def _hook_activations(self, module, inputs, outputs):
        """记录某一层layer输出的feature map"""
        self.activations = outputs

    def grads(self, outputs, inputs, retain_graph=True, create_graph=True):
        """默认保留梯度图，同时构建导数图（供计算高阶导数使用）"""
        grads = torch.autograd.grad(outputs=outputs,
                                    inputs=inputs,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)[0]
        grads = torch.sum(grads, dim=(2, 3))  # [b, c, h, w] => [b, c]
        self.model.zero_grad()

        return grads
    
model_name = "resnet32"
model_path = "/nfs/xwx/model-doctor-xwx/output/model/pretrained/resnet32/cifar-10-lt-ir100/lr0.01/cosine_lr_scheduler/ce_loss/2022-07-15_17-27-58/best-model-acc0.7144.pth"
data_name  = "cifar-10-lt-ir100"

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

data_loaders, _ = loaders.load_data(data_name=data_name)

model = models.load_model(
    model_name=model_name, 
    in_channels=3,
    num_classes=10
)

model.load_state_dict(torch.load(model_path)["model"])
model.to(device)
module = model.layer3[4].conv2

hook_module = HookModule(model, module)

# x = torch.rand(2, 3, 32, 32)
# labels = torch.tensor([1, 3])
# preds, _ = model(x)
# nll_loss = torch.nn.NLLLoss()(preds, labels)
# act = hook_module.activations
# grads = hook_module.grads(-nll_loss, act)

masks = np.load("/nfs/xwx/model-doctor-xwx/output/result/channels/resnet32-cifar-10-lt-ir100-ori/channels_-1.npy")
# masks = torch.from_numpy(np.asarray(masks))
# # print(masks)

# cl = 0
# for i, grad in enumerate(grads):
#     label = labels[i]  # 真实标签
#     mask = 1 - masks[label]  # 该类别对应的mask，并取反
#     cl = cl + torch.mean(torch.abs(grad * mask))
#     print(cl)
    

def get_channel_loss(logits, labels, channel_mask, hook_module: HookModule):
    nll_loss = torch.nn.NLLLoss()(logits, labels)
    act = hook_module.activations
    grads = hook_module.grads(-nll_loss, act)

    masks = torch.from_numpy(np.asarray(channel_mask)).to(device)

    cl = 0
    idx = logits.argmax(1) == labels
    grads = grads[idx] # 预测错误的梯度
    labels = labels[idx]  # 预测错误的标签
    # print(grads.shape)
    for i, grad in enumerate(grads):
        label = labels[i]
        mask = 1 - masks[label]
        cl = cl + torch.sum(torch.relu(grad * mask))

    return cl

def test(dataloader, model, device):
    size = len(dataloader.dataset)
    model.eval()
    correct = 0
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)
        pred, _ = model(X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
        
    print(f"Test Error: Accuracy: {(100*correct):>0.2f}%")
    
    return correct
    

def train(dataloader, model, loss_fn, optimizer, print_freq, device):
    train_loss, correct = 0, 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(True):
            pred, _ = model(X)  # 网络前向计算

            loss = loss_fn(pred, y)
            cl = get_channel_loss(pred, y, masks, hook_module)
            
            loss = loss + cl
            train_loss += loss.item()
            
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # Backpropagation
            optimizer.zero_grad()  # 清除过往梯度
            loss.backward()  # 得到模型中参数对当前输入的梯度
            optimizer.step()  # 更新参数
            
        if batch % print_freq == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"[train | {current:>5d}/{size:>5d}] loss: {loss:>7f}, cl loss: {cl:>7f}", flush=True)
    
    train_loss /= num_batches
    correct /= size
    
    print(f"Train Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {train_loss:>8f}")
    

base_lr = 0.01
total_epoch_num = 10
weight_decay = 5e-4 #weight decay value

optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch_num, eta_min=0.0)
loss_fn = nn.CrossEntropyLoss()
print_freq = 40

for epoch in range(total_epoch_num):
    print(f"\nEpoch {epoch+1}")
    train(data_loaders["train"], model, loss_fn, optimizer, print_freq, device)
    test(data_loaders["val"], model, device)
    scheduler.step()