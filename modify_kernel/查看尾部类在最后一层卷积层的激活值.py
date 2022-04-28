# 查看尾部类经过最后一层卷积层之后的激活值

import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

import torch
import torchvision
import models
import argparse
import os
import time
import yaml
import loaders
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.time_util import print_time
from modify_kernel.util.cfg_util import print_yml_cfg
from utils.args_util import print_args
from PIL import Image
import numpy as np
import seaborn as sns



parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='cifar-10-lt-ir100')
parser.add_argument('--tail_class', type=int, default='9')


def main():
    args = parser.parse_args()
    print_args(args)

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n[INFO] train on ', device)

    data_name = args.data_name
    cfg_filename = "cbs_refl.yml"
    cfg = get_cfg(cfg_filename)[data_name]
    print_yml_cfg(cfg)

    model_name = cfg["model_name"]
    # model_path = cfg["two_stage_model_path"]
    model_path = "/nfs/xwx/model-doctor-xwx/output/model/three-stage/resnet32/cifar-10-lt-ir100/lr0.1/th0.5/custom_lr_scheduler/refl_loss/2022-04-26_20-25-01/best-model-20220426-210920-acc0.7288.pth"
    

    # model
    model = models.load_model(
        model_name=model_name,
        in_channels=cfg['model']['in_channels'],
        num_classes=cfg['model']['num_classes']
    )

    model.load_state_dict(torch.load(model_path)["model"])
    model.to(device)

    # img_path = "/nfs/xwx/model-doctor-xwx/images/grad_percnet.png"
    # img = Image.open(img_path).convert("RGB")
    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((640, 640)),
    #     torchvision.transforms.ToTensor(),
    # ])
    # img_tensor = transforms(img).unsqueeze(dim=0).to(device)
    # img_label = torch.tensor(1).to(device)
    
    data_loaders, _ = loaders.load_data(data_name=data_name)

    tail_featuremaps = torch.zeros((64, 8, 8))
    
    begin_time = time.time()

    train(data_loaders["train"], model, device, tail_featuremaps, args)
    draw_tail_data_featuremaps(tail_featuremaps)
        
    print("Done!")
    print_time(time.time()-begin_time)


def train(dataloader, model,  device, tail_featuremaps, args):
    size = len(dataloader.dataset)
    model.eval()
    
    for batch, (X, y, _) in enumerate(dataloader):
        if batch % 10 == 0:
            current = batch * len(X)
            print(f"[{current:>5d}/{size:>5d}]", flush=True)

        X, y = X.to(device), y.to(device)

        X = X[y==args.tail_class]  # 尾部类样本
        if X.shape[0] == 0:
            continue

        _, featuremaps = model(X)  # 网络前向计算
        # print(featuremaps.shape)

        tail_featuremaps += featuremaps.detach().cpu().mean(dim=0)  # 64*8*8


def draw_tail_data_featuremaps(tail_featuremaps):
    tail_featuremaps = tail_featuremaps.numpy()
    plt.subplots(figsize=(64, 64), ncols=1)
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        sns.heatmap(tail_featuremaps[i], annot=False, xticklabels=False, yticklabels=False, cbar=False)
        plt.title(f"kernel {i}")

    plt.subplots_adjust(wspace=0.01, hspace=0.05)
    plt.savefig("tail_data_featuremap.png", bbox_inches='tight')
    plt.clf()
    plt.close()

           
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