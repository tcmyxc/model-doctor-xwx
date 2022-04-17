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
from hooks.grad_hook import GradHookModule
from modify_kernel.util.draw_util import draw_lr
from functools import partial

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='imagenet-10-lt')
parser.add_argument('--threshold', type=float, default='0.5')
parser.add_argument('--data_loader_type', type=int, default='0')

# global config
modify_dicts = []
# kernel_percents = {}
# threshold = None
best_acc = 0
best_model_path = None
result_path = None
g_train_loss, g_train_acc = [], []
g_test_loss, g_test_acc = [], []
g_cls_test_acc = {}

def main():
    args = parser.parse_args()
    print(f"\n[INFO] args: {args}")

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n[INFO] train on ', device)

    # get cfg
    global result_path, g_cls_test_acc

    data_name = args.data_name
    cfg_filename = "cbs_refl.yml"
    cfg = get_cfg(cfg_filename)[data_name]

    print("-" * 42)
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("-" * 42)

    for idx in range(cfg['model']['num_classes']):
        g_cls_test_acc[idx] = []

    model_name = cfg["model_name"]
    model_path = cfg["two_stage_model_path"]
    model_layers = range(cfg["model_layers"])

    # result path
    result_path = os.path.join(config.output_model, "three-stage",
                               model_name, data_name, 
                               f"lr{args.lr}", f"th{args.threshold}",
                               get_current_time())
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    print(f"\n[INFO] result will save in:\n{result_path}\n")

    # kernel
    num_classes = cfg['model']['num_classes']
    kernel_dict_path = os.path.join(
        cfg["kernel_dict_path"],
        f"{model_name}-{data_name}"
    )
    # 01mask
    for cls in range(num_classes):
        mask_path_patten = f"{kernel_dict_path}/kernel_dict_label_{cls}.npy"
        modify_dict = np.load(mask_path_patten, allow_pickle=True).item()
        modify_dicts.append(modify_dict)

    # data loader
    if args.data_loader_type == 0:
        # 常规数据加载器
        data_loaders, _ = loaders.load_data(data_name=data_name)
    elif args.data_loader_type == 1:
        # 类平衡采样
        data_loaders, _ = loaders.load_class_balanced_data(data_name=data_name)

    # model
    model = models.load_model(
        model_name=model_name,
        in_channels=cfg['model']['in_channels'],
        num_classes=cfg['model']['num_classes']
    )

    # modules
    modules = models.load_modules(
        model=model,
        model_name=model_name,
        model_layers=model_layers
    )

    model.load_state_dict(torch.load(model_path)["model"])
    model.to(device)
    
    begin_time = time.time()

    train(data_loaders["train"], model, modules, device)

    print("Done!")
    print_time(time.time()-begin_time)


def train(dataloader, model, loss_fn, optimizer, modules, epoch, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y, _) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        with torch.set_grad_enabled(True):
            # Compute prediction error
            pred, feature_out = model(X)  # 网络前向计算

            cal_cluster_center(X, y, pred, feature_out)
                

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"[{current:>5d}/{size:>5d}]", flush=True)


def cal_cluster_center(X, y, pred, feature_out):
    for cls, modify_dict in enumerate(modify_dicts):
        # 找到对应类别的图片
        x_pos = (y==cls).nonzero().squeeze()
        # 处理只有一个样本的情况
        if x_pos.shape == torch.Size([]):
            x_pos = x_pos.unsqueeze(dim=0)
        # 处理没有样本的情况
        if min(x_pos.shape) == 0:
            continue

        ft_cls_i = torch.index_select(feature_out, dim=0, index=x_pos)

        # 不相关卷积核的特征图往相关卷积核的特征图靠近
        layer = 29
        ft_err, ft_true = torch.zeros_like(ft_cls_i[:, 0, ::]), torch.zeros_like(ft_cls_i[:, 0, ::])
        for kernel_index in range(modify_dict[layer][0]):
            if kernel_index not in modify_dict[layer][1]:
                ft_err += ft_cls_i[:, kernel_index, ::]
            else:
                ft_true += ft_cls_i[:, kernel_index, ::]
            
        ft_loss += torch.abs(ft_err - ft_true).mean().item()
                        

def get_cfg(cfg_filename):
    """获取配置"""
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    # 获取当前文件所在目录
    curPath = os.path.dirname(os.path.realpath(__file__))
    # 获取yaml文件路径
    yamlPath = os.path.join(curPath, "config", cfg_filename)

    with open(yamlPath, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader)
    
    return cfg


if __name__ == '__main__':
    main()