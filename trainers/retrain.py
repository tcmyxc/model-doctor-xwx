"""
此脚本是在预训练模型的基础上, 使用相同的配置对模型重新训练, 查看带来的性能提升
"""

import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

import torch
import models
import loaders
import argparse
import os
import time

from torch import optim
from configs import config
from configs.config_util import get_cfg
from utils.lr_util import get_lr_scheduler
from utils.time_util import print_time, get_current_time
from sklearn.metrics import classification_report
from loss.refl import reduce_equalized_focal_loss
from loss.fl import focal_loss
from loss.bsl import balanced_softmax_loss
from loss.cbl import CB_loss
from modify_kernel.util.draw_util import draw_lr, draw_acc_and_loss, \
    draw_classification_report, draw_fc_weight, draw_fc_weight_history
from modify_kernel.util.cfg_util import print_yml_cfg
from functools import partial
from utils.args_util import print_args
from utils.general import update_best_model

import torch.nn as nn
from copy import deepcopy
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='cifar-100-lt-ir100')
parser.add_argument('--model_name', default='resnet32')
parser.add_argument('--threshold', type=float, default='0.5')
parser.add_argument('--lr', type=float, default='1e-2')
parser.add_argument('--data_loader_type', type=int, default='0', help='0 is default, 1 for cbs')
parser.add_argument('--epochs', type=int, default='200')
parser.add_argument('--lr_scheduler', type=str, default='cosine', help="choose from ['cosine', 'custom', 'constant']")
parser.add_argument('--loss_type', type=str, default='ce', help="choose from ['ce', 'fl', 'refl', 'bsl', 'cbl']")
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--model_path', type=str)


def main():
    args = parser.parse_args()
    print_args(args)

    # get cfg
    data_name    = args.data_name
    model_name   = args.model_name
    model_path   = args.model_path
    cfg_filename = "one_stage.yml"
    cfg = get_cfg(cfg_filename)[data_name]
    print_yml_cfg(cfg)

    # result path
    result_path = os.path.join(config.model_retrain,
                               model_name, data_name,
                               f"lr{args.lr}", f"{args.lr_scheduler}_lr_scheduler", 
                               f"{args.loss_type}_loss",
                               get_current_time())
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    print(f"\n[INFO] result will save in:\n{result_path}\n")
    
    # add some cfg
    cfg["best_model_path"] = None
    cfg["result_path"] = result_path
    cfg["best_acc"] = 0
    cfg["g_train_loss"] = []
    cfg["g_train_acc"] = []
    cfg["g_test_loss"] = []
    cfg["g_test_acc"] = []
    cfg["fc_weights"] = []

    lr = float(args.lr)
    momentum = cfg["optimizer"]["momentum"]
    weight_decay = float(cfg["optimizer"]["weight_decay"])
    epochs = args.epochs
    print(f"\n[INFO] total epoch: {epochs}")

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-' * 42, '\n[Info] use device:', device)

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
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"]
    )
    
    # 加载预训练模型
    model.load_state_dict(torch.load(model_path)["model"])
    model.to(device)

    # loss
    if args.loss_type == "ce":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_type == "fl":
        loss_fn = focal_loss
    elif args.loss_type == "refl":
        loss_fn = partial(reduce_equalized_focal_loss, threshold=args.threshold)
    elif args.loss_type == "bsl":
        sample_per_class = np.load(cfg["sample_per_class_path"])
        loss_fn = partial(balanced_softmax_loss, sample_per_class=sample_per_class)
    elif args.loss_type == "cbl":
        sample_per_class = np.load(cfg["sample_per_class_path"])
        loss_fn = partial(CB_loss,
                          samples_per_cls=sample_per_class, 
                          no_of_classes=cfg["model"]["num_classes"], 
                          loss_type="softmax", 
                          beta=0.9999, 
                          gamma=2, 
                          device=device
                          )
    
    # optimizer
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # lr scheduler
    if args.lr_scheduler == "custom":
        scheduler = get_lr_scheduler(optimizer, True)
    elif args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs
        )
    elif args.lr_scheduler == "constant":
        scheduler = optim.lr_scheduler.ConstantLR(
            optimizer=optimizer,
            factor=1,
            total_iters=epochs
        )


    begin_time = time.time()
    lr_list=[]
    for epoch in range(epochs):
        epoch_begin_time = time.time()
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        lr_list.append(cur_lr)
        print(f"\nEpoch {epoch+1}")
        print("[INFO] lr is:", cur_lr)
        print("-" * 42)

        train(data_loaders["train"], model, loss_fn, optimizer, device, cfg)
        test(data_loaders["val"], model, loss_fn, optimizer, epoch, device, args, cfg)
        scheduler.step()

        draw_acc_and_loss(cfg["g_train_loss"], cfg["g_test_loss"], cfg["g_train_acc"], cfg["g_test_acc"], result_path)
        draw_lr(result_path, lr_list)

        print_time(time.time()-epoch_begin_time, epoch=True)
        
    print("Done!")
    print_time(time.time()-begin_time)


def train(dataloader, model, loss_fn, optimizer, device, cfg):
    train_loss, correct = 0, 0
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(True):
            # Compute prediction error
            pred, _ = model(X)  # 网络前向计算

            loss = loss_fn(pred, y)
            train_loss += loss.item()
        
            y_pred_list.extend(pred.argmax(1).cpu().numpy())

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # Backpropagation
            optimizer.zero_grad()  # 清除过往梯度
            loss.backward()  # 得到模型中参数对当前输入的梯度
            optimizer.step()  # 更新参数

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"train | loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)
    
    train_loss /= num_batches
    correct /= size
    cfg["g_train_loss"].append(train_loss)
    cfg["g_train_acc"].append(correct)
    print("-" * 42)
    print(classification_report(y_train_list, y_pred_list, digits=4))


def test(dataloader, model, loss_fn, optimizer, epoch, device, args, cfg):
    loss_fn = nn.CrossEntropyLoss()
    model_name = args.model_name
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)
        with torch.set_grad_enabled(True):
            pred, _ = model(X)
            loss = loss_fn(pred, y)

            test_loss += loss.item()

        y_pred_list.extend(pred.argmax(1).cpu().numpy())

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"val | loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)

    test_loss /= num_batches
    cfg["g_test_loss"].append(test_loss)
    correct /= size
    cfg["g_test_acc"].append(correct)

    fc_weight = deepcopy(model.linear.weight.detach().cpu())
    tmp = torch.linalg.norm(fc_weight, ord=2, dim=1).detach().numpy()
    cfg["fc_weights"].append(tmp)
    
    if correct > cfg["best_acc"]:
        cfg["best_acc"] = correct
        print(f"\n[FEAT] Epoch {epoch+1}, update best acc:", correct)
        model_name=f"best-model-acc{correct:.4f}.pth"
        model_state = {
            'epoch': epoch,  # 注意这里的epoch是从0开始的
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': correct,
        }
        update_best_model(cfg, model_state, model_name)
        
        # 可视化最好的epoch分类头权重
        draw_fc_weight(cfg["result_path"], fc_weight.numpy())
        
    print(f"\nTest Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    print(classification_report(y_train_list, y_pred_list, digits=4))
    
    draw_classification_report("test", cfg["result_path"], y_train_list, y_pred_list)
    draw_fc_weight_history(cfg["result_path"], cfg["fc_weights"])



if __name__ == '__main__':
    main()