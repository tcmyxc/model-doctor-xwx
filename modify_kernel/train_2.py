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
from trainers.cls_trainer import print_time

from configs import config
import json

# 在导入matplotlib库后，且在matplotlib.pyplot库被导入前加下面这句话，不然不起作用
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 此脚本用于微调模型

lr = 1e-5
threshold = 0.5


modify_dicts = []
best_acc = 0
g_train_loss, g_train_acc = [], []
g_test_loss, g_test_acc = [], []

def main():
    # cfg
    data_name = 'cifar-10-lt-ir100'
    model_name = 'resnet32'
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 100
    model_layers = range(0, 30)
    
    cfg = json.load(open('../configs/config_trainer.json'))[data_name]

    num_classes=cfg['model']['num_classes']
    for cls in range(num_classes):
        mask_path_patten = f"/nfs/xwx/model-doctor-xwx/modify_kernel/kernel_dict/kernel_dict_label_{cls}.npy"
        modify_dict = np.load(mask_path_patten, allow_pickle=True).item()
        modify_dicts.append(modify_dict)

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-' * 79, '\n[Info] train on ', device)

    # data
    data_loaders, _ = loaders.load_data(data_name=data_name)

    # model
    model = models.load_model(
        model_name=model_name,
        in_channels=cfg['model']['in_channels'],
        num_classes=num_classes
    )

    modules = models.load_modules(
        model=model,
        model_name=model_name,
        model_layers=model_layers
    )
    
    cp_path = os.path.join('/nfs/xwx/model-doctor-xwx/output/model/pretrained/resnet32-cifar-10-lt-ir100-refl-th-0.4-wr/checkpoint.pth')
    if not os.path.exists(cp_path):
        print("=" * 40)
        print("模型文件的路径不存在, 请检查")
        return
    state = torch.load(cp_path)
   
    model.load_state_dict(state['model'])
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
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer=optimizer,
    #     T_max=epochs
    # )

    for epoch in range(epochs):
        epoch_begin_time = time.time()
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        print(f"\nEpoch {epoch+1}")
        print("[INFO] lr is:", cur_lr)
        print("-"*40)
        train(data_loaders["train"], model, loss_fn, optimizer, modules, device)
        test(data_loaders["val"], model, loss_fn, optimizer, epoch, device)
        scheduler.step()

        draw_acc(g_train_loss, g_test_loss, g_train_acc, g_test_acc)
        print_time(time.time()-epoch_begin_time, epoch=True)
        
    print("Done!")

    # model.eval()
    # test(data_loaders["val"], model, loss_fn, device)


def train(dataloader, model, loss_fn, optimizer, modules, device):
    global g_train_loss, g_train_acc
    train_loss, correct = 0, 0
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
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
                    # if layer <= 19:
                    #     modules[int(layer)].weight.grad[:] = 0
                    # # # print("layer:", layer)
                    for kernel_index in range(modify_dict[layer][0]):
                        if kernel_index not in modify_dict[layer][1]:
                            modules[int(layer)].weight.grad[kernel_index, ::] = 0
            

                optimizer.step()  # 更新参数

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= (num_batches * len(modify_dicts))
    correct /= size
    g_train_loss.append(train_loss)
    g_train_acc.append(correct)
    print("-" * 40)
    print(classification_report(y_train_list, y_pred_list, digits=4))


def test(dataloader, model, loss_fn, optimizer, epoch, device):
    global best_acc, g_test_loss, g_test_acc
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    for X, y in dataloader:
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
        print("[FEAT] update best acc:", best_acc)

        best_model_name=f"best-model-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-acc{best_acc:.4f}.pth"
        model_state = {
            'epoch': epoch,  # 注意这里的epoch是从0开始的
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': best_acc,
        }
        torch.save(model_state, best_model_name)

        print(f"Saved Best PyTorch Model State to {best_model_name} \n")
    print(f"Test Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    print(classification_report(y_train_list, y_pred_list, digits=4))


def draw_acc(train_loss, test_loss, train_acc, test_acc):
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
        plt.savefig('model_lr1e-5.jpg')
        plt.clf()
        plt.close()


if __name__ == '__main__':
    main()