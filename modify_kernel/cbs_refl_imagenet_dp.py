import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

from loss.refl import reduce_equalized_focal_loss

import torch
import models
import loaders
import argparse
import os
import datetime
import time
import matplotlib
import yaml

from torch import optim
from utils.lr_util import get_lr_scheduler
from sklearn.metrics import classification_report
from utils.time_util import print_time
from configs import config

# 在导入matplotlib库后，且在matplotlib.pyplot库被导入前加下面这句话，不然不起作用
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# 使用类别平衡采样和REFL对预训练模型进行调整

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='imagenet-lt')
parser.add_argument('--threshold', type=float, default='0.4')

threshold = None
best_acc = 0
best_model_path = None
g_train_loss, g_train_acc = [], []
g_test_loss, g_test_acc = [], []

def main():
    args = parser.parse_args()
    print(f"\n[INFO] args: {args} \n")

    # device
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('-' * 79, '\n[INFO] train on ', device)

    # get cfg
    global threshold
    threshold = args.threshold
    
    data_name = args.data_name
    cfg_filename = "imagenet_lt.yml"
    cfg = get_cfg(cfg_filename)[data_name]

    print("-" * 42)
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("-" * 42)

    model_name = cfg["model_name"]
    pretrained_model_path = cfg["pretrained_model_path"]
    lr = float(cfg["optimizer"]["lr"])
    momentum = cfg["optimizer"]["momentum"]
    weight_decay = float(cfg["optimizer"]["weight_decay"])
    epochs = cfg["epochs"]

    # data
    data_loaders, _ = loaders.load_data(data_name=data_name)
    if "cifar" in data_name:
        print("\n[INFO] use cbs sampler")
        data_loaders, _ = loaders.load_class_balanced_data(data_name=data_name)

    # model
    model = models.load_model(
        model_name=model_name,
        in_channels=cfg['model']['in_channels'],
        num_classes=cfg['model']['num_classes']
    )
    model.load_state_dict(torch.load(pretrained_model_path)["model"])
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("\n[INFO] use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[1, 2, 3])

    # optimizer
    loss_fn = reduce_equalized_focal_loss
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = get_lr_scheduler(optimizer, True)

    begin_time = time.time()
    for epoch in range(epochs):
        epoch_begin_time = time.time()
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        print(f"\nEpoch {epoch+1}")
        print("[INFO] lr is:", cur_lr)
        print("-"*40)
        train(data_loaders["train"], model, loss_fn, optimizer, device)
        test(data_loaders["val"], model, loss_fn, optimizer, epoch, device, args, cfg)
        scheduler.step()

        draw_acc(g_train_loss, g_test_loss, g_train_acc, g_test_acc, args, cfg)
        print_time(time.time()-epoch_begin_time, epoch=True)
        
    print("Done!")
    print_time(time.time()-begin_time)


def train(dataloader, model, loss_fn, optimizer, device):
    global g_train_loss, g_train_acc
    train_loss, correct = 0, 0
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    size = len(dataloader.dataset)
    # size = 50000
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y, _) in enumerate(dataloader):
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(True):
            # Compute prediction error
            pred = model(X)  # 网络前向计算
            loss = loss_fn(pred, y, threshold=threshold)

            train_loss += loss.item()
        
            y_pred_list.extend(pred.argmax(1).cpu().numpy())

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()  # 得到模型中参数对当前输入的梯度
            optimizer.step()  # 更新参数
    
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)
    
    train_loss /= num_batches
    correct /= size
    g_train_loss.append(train_loss)
    g_train_acc.append(correct)
    print("-" * 40)
    print(classification_report(y_train_list, y_pred_list, digits=4))


def test(dataloader, model, loss_fn, optimizer, epoch, device, args, cfg):
    model_name = cfg["model_name"]
    data_name = args.data_name
    result_path = os.path.join(config.output_model,
                               "cbs-refl",
                               f"{model_name}-{data_name}-th{args.threshold}")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    global best_acc, g_test_loss, g_test_acc
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    for X, y, _ in dataloader:
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)
        with torch.set_grad_enabled(True):
            pred = model(X)
            loss = loss_fn(pred, y, threshold=threshold)

            test_loss += loss.item()

        y_pred_list.extend(pred.argmax(1).cpu().numpy())

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    g_test_loss.append(test_loss)
    correct /= size
    g_test_acc.append(correct)
    if correct > best_acc:
        best_acc = correct
        print(f"[FEAT] epoch {epoch+1}, update best acc: {best_acc:.4f}")
        model_name=f"best-model-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-acc{best_acc:.4f}.pth"
        model_state = {
            'epoch': epoch,  # 注意这里的epoch是从0开始的
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': best_acc,
        }
        update_best_model(result_path, model_state, model_name)

    print(f"Test Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    print(classification_report(y_train_list, y_pred_list, digits=4))


def draw_acc(train_loss, test_loss, train_acc, test_acc, args, cfg):
    model_name = cfg["model_name"]
    data_name = args.data_name
    history = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': test_loss,
        'val_acc': test_acc
    }
    
    result_path = os.path.join(config.output_model,
                               "cbs-refl",
                               f"{model_name}-{data_name}-th{args.threshold}")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    np.save(os.path.join(result_path, 'model.npy'), history)

    num_epochs = len(train_loss)

    plt.plot(range(1, num_epochs + 1), train_loss, 'r', label='train loss')
    plt.plot(range(1, num_epochs + 1), test_loss, 'b', label='val loss')

    plt.plot(range(1, num_epochs + 1), train_acc, 'g', label='train acc')
    plt.plot(range(1, num_epochs + 1), test_acc, 'k', label='val acc')

    plt.title("Acc and Loss of each epoch")
    plt.xlabel("Training Epochs")
    plt.ylabel("Acc & Loss")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(result_path, "cbs_refl_model.jpg"))
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


def update_best_model(result_path, model_state, model_name):
    """更新权重文件"""
    global best_model_path
    cp_path = os.path.join(result_path, model_name)

    if best_model_path is not None:
        # remove previous model weights
        os.remove(best_model_path)

    torch.save(model_state, cp_path)
    best_model_path = cp_path
    print(f"Saved Best PyTorch Model State to {model_name} \n")


if __name__ == '__main__':
    main()