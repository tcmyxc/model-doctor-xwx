# 此脚本只是单纯训练一个分类模型
# 1. 使用普通的数据加载器
# 2. 使用REFL

import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

import os
import time
import yaml
import torch
import models
import loaders
import datetime
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from torch import optim
from sklearn.metrics import classification_report
from utils.lr_util import get_lr_scheduler
from utils.time_util import print_time
from loss.refl import reduce_equalized_focal_loss



parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='imagenet-10-lt')
parser.add_argument('--model_name', default='resnext50')
parser.add_argument('--threshold', type=float, default='0.5')
parser.add_argument('--loss_name', type=str, default='ce')


best_acc = 0
best_model_path = None
g_train_loss, g_train_acc = [], []
g_test_loss, g_test_acc = [], []

def main():
    args = parser.parse_args()
    print(f"\n[INFO] args: {args} \n")

    # get cfg
    data_name   = args.data_name
    model_name  = args.model_name
    cfg_filename = "one_stage.yml"
    cfg = get_cfg(cfg_filename)[data_name]

    print("-" * 42)
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("-" * 42)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-' * 42, '\n[Info] train on ', device)

    # data,普通数据加载器
    data_loaders, _ = loaders.load_data(data_name=data_name)

    # model
    model = models.load_model(
        model_name=model_name, 
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"]
    )
    model.to(device)

    loss_fn = reduce_equalized_focal_loss
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=cfg["optimizer"]["lr"],
        momentum=cfg["optimizer"]["momentum"],
        weight_decay=float(cfg["optimizer"]["weight_decay"])
    )
    # lr scheduler
    # scheduler = get_lr_scheduler(optimizer, True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cfg["epochs"]
    )

    begin_time = time.time()
    lr_list=[]
    for epoch in range(cfg["epochs"]):
        epoch_begin_time = time.time()
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        lr_list.append(cur_lr)
        print(f"\nEpoch {epoch+1}")
        print("[INFO] lr is:", cur_lr)
        print("-" * 42)
        train(data_loaders["train"], model, loss_fn, optimizer, device, args)
        test(data_loaders["val"], model, loss_fn, optimizer, epoch, device, args)
        scheduler.step()

        draw_acc(g_train_loss, g_test_loss, g_train_acc, g_test_acc, args)
        draw_lr(lr_list)
        print_time(time.time()-epoch_begin_time, epoch=True)
        
    print("Done!")
    print_time(time.time()-begin_time)


def train(dataloader, model, loss_fn, optimizer, device, args):
    threshold  = args.threshold
    global g_train_loss, g_train_acc
    train_loss, correct = 0, 0
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y, _) in enumerate(dataloader):
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(True):
            # Compute prediction error
            pred = model(X)  # 网络前向计算
            loss = loss_fn(pred, y, threshold=threshold)
            # loss = focal_loss(pred, y)

            train_loss += loss.item()
        
            y_pred_list.extend(pred.argmax(1).cpu().numpy())

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # Backpropagation
            optimizer.zero_grad()  # 清除过往梯度
            loss.backward()  # 得到模型中参数对当前输入的梯度
            optimizer.step()  # 更新参数

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)
    
    train_loss /= num_batches
    correct /= size
    g_train_loss.append(train_loss)
    g_train_acc.append(correct)
    print("-" * 42)
    print(classification_report(y_train_list, y_pred_list, digits=4))


def test(dataloader, model, loss_fn, optimizer, epoch, device, args):
    data_name  = args.data_name
    model_name = args.model_name
    threshold  = args.threshold
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
            # loss = focal_loss(pred, y)

            test_loss += loss.item()

        y_pred_list.extend(pred.argmax(1).cpu().numpy())

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    g_test_loss.append(test_loss)
    correct /= size
    g_test_acc.append(correct)
    if correct > best_acc:
        best_acc = correct
        print(f"[FEAT] Epoch {epoch+1}, update best acc:", best_acc)
        model_name=f"best-model-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-acc{best_acc:.4f}.pth"
        model_state = {
            'epoch': epoch,  # 注意这里的epoch是从0开始的
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': best_acc,
        }
        update_best_model("./pretained", model_state, model_name)
        
    print(f"Test Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    print(classification_report(y_train_list, y_pred_list, digits=4))
    draw_classification_report("test", "./pretained", y_train_list, y_pred_list)


def draw_acc(train_loss, test_loss, train_acc, test_acc, args):
    data_name  = args.data_name
    model_name = args.model_name
    threshold  = args.threshold
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
    plt.savefig(f'{model_name}_{data_name}_th{threshold}_pure_train_cls_model.jpg')
    plt.clf()
    plt.close()

def draw_lr(lr_list):
    num_epochs = len(lr_list)

    plt.plot(range(1, num_epochs + 1), lr_list, label='lr')

    plt.title("Learning rate of each epoch")
    plt.xlabel("Training Epochs")
    plt.ylabel("Learning rate")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.legend()
    plt.savefig('model_lr.jpg')
    plt.clf()
    plt.close()


def update_best_model(result_path, model_state, model_name):
    """更新权重文件"""
    global best_model_path
    cp_path = os.path.join(result_path, model_name)
    # cp_path = model_name

    if best_model_path is not None:
        # remove previous model weights
        os.remove(best_model_path)

    torch.save(model_state, cp_path)
    best_model_path = cp_path
    print(f"Saved Best PyTorch Model State to {model_name} \n")


def draw_classification_report(mode_type, result_path, y_train_list, y_pred_list):
    """绘制在训练集/测试集上面的 classification_report"""
    reports = classification_report(y_train_list, y_pred_list, digits=4, output_dict=True)
    np.save(os.path.join(result_path, f"{mode_type}_classification_report.npy"), reports)

    labels = []
    accs = []
    samplers =[]
    for x_i, y_i in reports.items():
        if x_i == "accuracy": break
        labels.append(x_i)
        accs.append(y_i["recall"])
        samplers.append(y_i["support"])

    plt.plot(labels, accs)
    plt.title("Acc of each class")
    plt.xlabel("Classes")
    plt.ylabel("Accuracy")
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



if __name__ == '__main__':
    main()