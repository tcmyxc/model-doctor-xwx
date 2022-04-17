import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

from loss.fl import focal_loss
from loss.efl import equalized_focal_loss
from loss.refl import reduce_equalized_focal_loss
from loss.rfl import reduced_focal_loss
from loss.dfl import dual_focal_loss

import torch
import torch.nn as nn
from torch import optim
import models
import loaders

from utils.lr_util import get_lr_scheduler
from sklearn.metrics import classification_report

from tqdm import tqdm
import os
import datetime

import numpy as np
import time

import matplotlib
from utils.time_util import print_time

from configs import config
import json

# 在导入matplotlib库后，且在matplotlib.pyplot库被导入前加下面这句话，不然不起作用
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default='1e-5')
parser.add_argument('--threshold', type=float, default='0.5')

# 计算loss使用常规方法

# lr = 1e-2
# threshold = 0.5
epochs = 100


mask_path = "/nfs/xwx/model-doctor-xwx/modify_kernel/kernel_dict/resnet32-cifar-10-lt-ir100/kernel_dict_label_789.npy"  # label_8_9
modify_dict = np.load(mask_path, allow_pickle=True).item()

print("-"*40)
for k, v in modify_dict.items():
    print(k, v)
print("-"*40)


best_acc = 0
best_model_path = None
g_train_loss, g_train_acc = [], []
g_test_loss, g_test_acc = [], []

def main():
    args = parser.parse_args()
    lr = float(args.lr)
    print(f"\n[INFO] args: {args} \n")
    # cfg
    data_name = 'cifar-10-lt-ir100'
    model_name = 'resnet32'
    momentum = 0.9
    weight_decay = 5e-4
    model_layers = range(0, 30)
    
    cfg = json.load(open('../configs/config_trainer.json'))[data_name]

    num_classes=cfg['model']['num_classes']

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-' * 79, '\n[Info] train on ', device)

    # data
    data_loaders, _ = loaders.load_class_balanced_data(data_name=data_name)

    # model
    model = models.load_model(
        model_name=model_name,
        in_channels=cfg['model']['in_channels'],
        num_classes=num_classes
    )

    # modules
    modules = models.load_modules(
        model=model,
        model_name=model_name,
        model_layers=model_layers
    )
    
    # checkpoint
    cp_path = os.path.join('/nfs/xwx/model-doctor-xwx/modify_kernel/消融实验/自定义学习率-lr1e-4-th-0.5-refl-cbs/best-model-20220303-121641-acc0.7757.pth')
    if not os.path.exists(cp_path):
        print("=" * 40)
        print("模型文件的路径不存在, 请检查")
        return
    state = torch.load(cp_path)
   
    model.load_state_dict(state)
    model.to(device)


    # cfg
    loss_fn = reduce_equalized_focal_loss
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = get_lr_scheduler(optimizer, True)

    for epoch in range(epochs):
        epoch_begin_time = time.time()
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        print(f"\nEpoch {epoch+1}")
        print("[INFO] lr is:", cur_lr)
        print("-"*40)
        train(data_loaders["train"], model, loss_fn, optimizer, modules, device, args)
        test(data_loaders["val"], model, loss_fn, optimizer, epoch, device, args)
        scheduler.step()

        draw_acc(g_train_loss, g_test_loss, g_train_acc, g_test_acc, args)
        print_time(time.time()-epoch_begin_time, epoch=True)
        
    print("Done!")


def train(dataloader, model, loss_fn, optimizer, modules, device, args):
    # threshold
    threshold = args.threshold

    global g_train_loss, g_train_acc
    train_loss, correct = 0, 0
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    # size = len(dataloader.dataset)
    size = 50000 # cifar
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
            
            for layer in modify_dict.keys():
                    # if layer <= 19:
                    #     modules[int(layer)].weight.grad[:] = 0
                    # # # print("layer:", layer)
                    for kernel_index in range(modify_dict[layer][0]):
                        if kernel_index in modify_dict[layer][1]:
                            modules[int(layer)].weight.grad[kernel_index, ::] = 0
            optimizer.step()  # 更新参数
    
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    correct /= size
    g_train_loss.append(train_loss)
    g_train_acc.append(correct)
    print("-" * 40)
    print(classification_report(y_train_list, y_pred_list, digits=4))


def test(dataloader, model, loss_fn, optimizer, epoch, device, args):
    # threshold
    threshold = args.threshold

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
    if correct >= best_acc:
        best_acc = correct
        print(f"[FEAT] epoch {epoch+1}, update best acc:", best_acc)

        model_name=f"best-model-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-acc{best_acc:.4f}.pth"
        model_state = {
            'epoch': epoch,  # 注意这里的epoch是从0开始的
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': best_acc,
        }
        update_best_model(None, model_state, model_name)
    print(f"Test Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    print(classification_report(y_train_list, y_pred_list, digits=4))


def draw_acc(train_loss, test_loss, train_acc, test_acc, args):
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
        plt.savefig(f'fine_tuning_lr{args.lr}_th{args.threshold}_v2.jpg')
        plt.clf()
        plt.close()


def update_best_model(result_path, model_state, model_name):
    """更新权重文件"""
    global best_model_path
    cp_path = model_name

    if best_model_path is not None:
        # remove previous model weights
        os.remove(best_model_path)

    torch.save(model_state, cp_path)
    torch.save(model_state, os.path.join(result_path, "best-model.pth"))
    best_model_path = cp_path
    print(f"Saved Best PyTorch Model State to {model_name} \n")
if __name__ == '__main__':
    main()
