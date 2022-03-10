from random import sample
import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

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
from configs import config
from utils.lr_util import get_lr_scheduler
from trainers.cls_trainer import print_time
from sklearn.metrics import classification_report
from loss.refl import reduce_equalized_focal_loss

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# 使用类别平衡采样和REFL对预训练模型进行调整

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='cifar-100-lt-ir100')
parser.add_argument('--threshold', type=float, default='0.5')


# global config
modify_dicts = []
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
    print('-' * 79, '\n[Info] train on ', device)

    # get cfg
    global threshold
    threshold = args.threshold
    data_name = args.data_name
    cfg_filename = "cbs_refl.yml"
    cfg = get_cfg(cfg_filename)[data_name]

    print("-" * 42)
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("-" * 42)

    model_name = cfg["model_name"]
    model_path = cfg["two_stage_model_path"]
    lr = float(cfg["optimizer"]["lr"])
    momentum = cfg["optimizer"]["momentum"]
    weight_decay = float(cfg["optimizer"]["weight_decay"])
    epochs = cfg["three_stage_epochs"]
    model_layers = range(cfg["model_layers"])

    # kernel
    num_classes = cfg['model']['num_classes']
    kernel_dict_path = os.path.join(
        cfg["kernel_dict_path"],
        f"{model_name}-{data_name}"
    )
    for cls in range(num_classes):
        mask_path_patten = f"{kernel_dict_path}/kernel_dict_label_{cls}.npy"
        modify_dict = np.load(mask_path_patten, allow_pickle=True).item()
        modify_dicts.append(modify_dict)
    
    # print(modify_dicts)

    # data
    data_loaders, _ = loaders.load_data(data_name=data_name)
    if "cifar" in data_name:
        print("\n[INFO] use cbs sampler \n")
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
    if torch.cuda.device_count() > 1:
        print("\n[INFO] Use", torch.cuda.device_count(), "GPUs! \n")
        model = nn.DataParallel(model, device_ids=[0, 1])

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
        print("-" * 42)

        train(data_loaders["train"], model, loss_fn, optimizer, modules, device, args)
        test(data_loaders["val"], model, loss_fn, optimizer, epoch, device, args, cfg)
        scheduler.step()

        draw_acc(g_train_loss, g_test_loss, g_train_acc, g_test_acc, args, cfg)
        print_time(time.time()-epoch_begin_time, epoch=True)
        
    print("Done!")
    print_time(time.time()-begin_time)


def train(dataloader, model, loss_fn, optimizer, modules, device, args):
    global g_train_loss, g_train_acc
    train_loss, correct = 0, 0
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    # size = len(dataloader.dataset)
    size = 50000  # cifar
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y, _) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        for cls, modify_dict in enumerate(modify_dicts):
            with torch.set_grad_enabled(True):
                # 找到对应类别的图片
                x_pos = (y==cls).nonzero().squeeze()
                # 处理只有一个样本的情况
                if x_pos.shape == torch.Size([]):
                    x_pos = x_pos.unsqueeze(dim=0)
                # 处理没有样本的情况
                if min(x_pos.shape) == 0:
                    continue
                x_cls_i = torch.index_select(X, dim=0, index=x_pos)
                y_cls_i = torch.index_select(y, dim=0, index=x_pos)
                y_train_list.extend(y_cls_i.cpu().numpy())
                # Compute prediction error
                pred = model(x_cls_i)  # 网络前向计算
                loss = loss_fn(pred, y_cls_i, threshold=threshold)
                # loss = focal_loss(pred, y)

                train_loss += loss.item()
            
                y_pred_list.extend(pred.argmax(1).cpu().numpy())

                correct += (pred.argmax(1) == y_cls_i).type(torch.float).sum().item()
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()  # 得到模型中参数对当前输入的梯度
            
                for layer in modify_dict.keys():
                    for kernel_index in range(modify_dict[layer][0]):
                        if kernel_index not in modify_dict[layer][1]:
                            modules[int(layer)].weight.grad[kernel_index, ::] = 0
            

                optimizer.step()  # 更新参数

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    correct /= size
    g_train_loss.append(train_loss)
    g_train_acc.append(correct)
    print("-" * 42)
    print(classification_report(y_train_list, y_pred_list, digits=4))


def test(dataloader, model, loss_fn, optimizer, epoch, device, args, cfg):
    model_name = cfg["model_name"]
    data_name = args.data_name
    result_path = os.path.join(config.output_model,
                               "three-stage",
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
    draw_classification_report("test", result_path, y_train_list, y_pred_list)


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
                               "three-stage",
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
    plt.savefig(os.path.join(result_path, "three_stage_model.jpg"))
    plt.clf()
    plt.close()


def draw_classification_report(mode_type, result_path, y_train_list, y_pred_list):
    """绘制在训练集/测试集上面的 classification_report"""
    reports = classification_report(y_train_list, y_pred_list, digits=4, output_dict=True)
    np.save(os.path.join(result_path, f"{mode_type}_classification_report.npy"), reports)

    labels = []
    accs = []
    samplers =[]
    for x_i, y_i in reports:
        labels.append(x_i)
        accs.append(y_i["recall"])
        samplers.append(y_i["support"])

    plt.plot(labels, accs)
    plt.title("Acc of each class")
    plt.xlabel("Classes")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(result_path, f"{mode_type}_classification_report.jpg"))
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