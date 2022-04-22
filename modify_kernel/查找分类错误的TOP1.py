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
from modify_kernel.util.cfg_util import print_yml_cfg
from functools import partial
from utils.args_util import print_args

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='cifar-10-lt-ir100')
parser.add_argument('--data_loader_type', type=int, default='0')

def main():
    args = parser.parse_args()
    print_args(args)

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n[INFO] train on {device}\n')

    data_name = args.data_name
    cfg_filename = "cbs_refl.yml"
    cfg = get_cfg(cfg_filename)[data_name]
    print_yml_cfg(cfg)

    model_name = cfg["model_name"]
    model_path = cfg["two_stage_model_path"]
    num_classes=cfg['model']['num_classes']

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

    model.load_state_dict(torch.load(model_path)["model"])
    model.to(device)

    train(data_loaders["train"], model, num_classes, device)
    test(data_loaders["val"], model, num_classes, device)


def train(dataloader, model, num_classes, device):
    y_pred_list = []
    y_train_list = []
    size = len(dataloader.dataset)
    model.eval()
    for batch, (X, y, _) in enumerate(dataloader):
        y_train_list.extend(y.numpy())
        X, y = X.to(device), y.to(device)
        
        pred, feature_out = model(X)  # 网络前向计算
        y_pred_list.extend(pred.argmax(1).cpu().numpy())


        if batch % 10 == 0:
            current = batch * len(X)
            print(f"train [{current:>5d}/{size:>5d}]", flush=True)

    draw_pred(num_classes, y_train_list, y_pred_list, mode="train")

def test(dataloader, model, num_classes, device):
    y_pred_list = []
    y_train_list = []
    
    size = len(dataloader.dataset)
   
    model.eval()
    for batch, (X, y, _) in enumerate(dataloader):
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)
        pred, feature_out = model(X)  # 网络前向计算

        y_pred_list.extend(pred.argmax(1).cpu().numpy())


        if batch % 10 == 0:
            current = batch * len(X)
            print(f"test [{current:>5d}/{size:>5d}]", flush=True)


    draw_pred(num_classes, y_train_list, y_pred_list, mode="val")

def draw_pred(num_classes, y_train_list, y_pred_list, mode="train"):
    # 10个类别有10个桶
    # 1个桶里面有10个小桶，初始化为0
    pred_dics = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for input, pred in zip(y_train_list, y_pred_list):
        # 先找大桶，再找小桶
        pred_dics[input][pred] += 1

    # print(pred_dics)
    for cls in range(num_classes):
        x = range(num_classes)
        # plt.plot(x, pred_dics[cls], label=f"cls {cls}")
        plt.plot(x, pred_dics[cls])
        for x0, y0 in zip(x, pred_dics[cls]):
            plt.text(x0, y0, f"{y0}", fontsize=9)
        plt.title(f"{mode} class {cls}")
        plt.xlabel("class")
        plt.ylabel("amount")
        plt.xticks(range(num_classes))
        plt.savefig(f"images/{mode}_class{cls}_pred.jpg")
        plt.clf()
        plt.close()
    
    # plt.xlabel("class")
    # plt.ylabel("amount")
    # plt.legend(loc="best")
    # plt.xticks(range(num_classes))
    # plt.savefig(f"{mode}_pred.jpg")
    # plt.clf()
    # plt.close()





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