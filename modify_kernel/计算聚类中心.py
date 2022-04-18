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
parser.add_argument('--data_name', default='cifar-10-lt-ir100')
parser.add_argument('--data_loader_type', type=int, default='0')

ft_centers = None

def main():
    args = parser.parse_args()
    print(f"\n[INFO] args: {args}")

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n[INFO] train on ', device)

    data_name = args.data_name
    cfg_filename = "cbs_refl.yml"
    cfg = get_cfg(cfg_filename)[data_name]

    print("-" * 42)
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("-" * 42)

    model_name = cfg["model_name"]
    model_path = cfg["two_stage_model_path"]
    num_classes = cfg['model']['num_classes']

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
        num_classes=num_classes
    )

    model.load_state_dict(torch.load(model_path)["model"])
    model.to(device)

    result_path = os.path.join(f"/nfs/xwx/model-doctor-xwx/output/result/{model_name}-{data_name}", 'features')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    begin_time = time.time()

    train(data_loaders["train"], model, num_classes, device)
    np.save(os.path.join(result_path, "ft_centers.npy"), ft_centers)
    # draw_tsne(result_path, ft_centers)

    print("Done!")
    print_time(time.time()-begin_time)


def train(dataloader, model, num_classes, device):
    global ft_centers
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    for batch, (X, y, _) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        with torch.set_grad_enabled(True):
            # Compute prediction error
            pred, feature_out = model(X)  # 网络前向计算
            
            # 挑选分类对的
            correct = pred.argmax(1) == y  # 分类正确的索引
            feature_out = feature_out[correct]
            y = y[correct]

            cal_cluster_center(y, feature_out, num_classes)
                

        if batch % 10 == 0:
            current =  batch * len(X)
            print(f"\r[{current:>5d}/{size:>5d}]", flush=True)
    
    ft_centers /= num_batches
    print(ft_centers.mean())


def cal_cluster_center(y, feature_out, num_classes):
    _, k, h, w = feature_out.shape
    global ft_centers
    if ft_centers is None:
        ft_centers = np.zeros((num_classes, k, h, w))
    
    # print(ft_centers.mean())

    for cls in range(num_classes):
        # 找到对应类别的图片
        x_pos = (y==cls).nonzero().squeeze()
        # 处理只有一个样本的情况
        if x_pos.shape == torch.Size([]):
            x_pos = x_pos.unsqueeze(dim=0)
        # 处理没有样本的情况
        if min(x_pos.shape) == 0:
            continue

        ft_cls_i = torch.index_select(feature_out, dim=0, index=x_pos)
        ft_cls_i = torch.mean(ft_cls_i, dim=0)
        ft_centers[cls] += ft_cls_i.detach().cpu().numpy()
                        

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


def draw_tsne(result_path, features):
    features = torch.flatten(torch.tensor(features), 1).numpy()
    # 特征图可视化
    from sklearn.manifold import TSNE
    # colorBoard=["dimgray","darkorange","tan","silver","forestgreen",\
    #             "darkgreen","royalblue","navy","red","darksalmon","peru","olive",\
    #             "yellow","cyan","mediumaquamarine","skyblue","purple","fuchsia",\
    #             "indigo","khaki"]

    colors = np.array(["C0","C1","C2","C3","C4","C5","C6","C7", "C8","C9"])
    # colors = np.array(colorBoard)

    features_embedded = TSNE(n_components=2, init='pca', n_iter=1000).fit_transform(features)
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=colors)
    # plt.colorbar()
    plt.savefig(os.path.join(result_path, f"tsne.jpg"))
    plt.clf()
    plt.close()

if __name__ == '__main__':
    main()