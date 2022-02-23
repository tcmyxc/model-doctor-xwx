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

# 在导入matplotlib库后，且在matplotlib.pyplot库被导入前加下面这句话，不然不起作用
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 梯度值
# modify_dict = {
#     # -1: [64, [4, 19, 20, 27, 28, 36, 38, 40, 46, 50, 53]],  # label_9, 11个
#     # -1: [64, [4, 19, 20, 27, 28, 36, 38, 40, 46, 50, 53, 0, 1, 18, 31, 33, 44, 52, 54, 55, 56, 62]],  # label_89, 22个
#     # -1: [64, [4, 19, 20, 27, 28, 36, 38, 40, 46, 50, 53, 
#     #           0, 1, 18, 31, 33, 44, 52, 54, 55, 56, 62,
#     #           2, 3, 5, 8, 10, 11, 12, 16, 22, 30, 32, 35, 39, 59, 60]],  # label_789, 37个
#     -1: [64, [3, 4, 5, 8, 10, 11, 12, 13, 17, 19, 29, 32, 35, 37, 39, 42, 47, 49, 53, 60,
#               2, 16, 20, 22, 30, 38, 56, 59,
#               27, 28, 36, 40, 46, 50]],  # 579, 34个
# }
mask_path = "/nfs/xwx/model-doctor-xwx/modify_kernel/kernel_dict/kernel_dict_label_89.npy"  # label_8&9
modify_dict = np.load(mask_path, allow_pickle=True).item()

print("-"*40)
for k, v in modify_dict.items():
    print(k, v)
print("-"*40)

threshold = 0.5
best_acc = 0
g_train_loss, g_train_acc = [], []
g_test_loss, g_test_acc = [], []

def main():
    # cfg
    data_name = 'cifar-10-lt-ir100'
    model_name = 'resnet32'
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 200
    model_layers = range(0, 30)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-' * 79, '\n[Info] train on ', device)

    # data
    data_loaders, _ = loaders.load_data(data_name=data_name)

    # model
    model = models.load_model(
        model_name=model_name,
        in_channels=3,
        num_classes=10
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
    # scheduler = get_lr_scheduler(optimizer, True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs
    )

    for t in range(epochs):
        epoch_begin_time = time.time()
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        print(f"\nEpoch {t+1}")
        print("[INFO] lr is:", cur_lr)
        print("-"*40)
        train(data_loaders["train"], model, loss_fn, optimizer, modules, device)
        test(data_loaders["val"], model, loss_fn, device)
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
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(True):
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y, threshold=threshold)

            train_loss += loss.item()
        
            y_pred_list.extend(pred.argmax(1).cpu().numpy())

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            for layer in modify_dict.keys():
                if layer <= 19:
                    continue
                # print("layer:", layer)
                for kernel_index in range(modify_dict[layer][0]):
                    if kernel_index not in modify_dict[layer][1]:
                        modules[int(layer)].weight.grad[kernel_index, ::] = 0
            

            optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    correct /= size
    g_train_loss.append(train_loss)
    g_train_acc.append(correct)
    print("-" * 40)
    print(classification_report(y_train_list, y_pred_list, digits=4))


def test(dataloader, model, loss_fn, device):
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
            test_loss += loss_fn(pred, y, threshold=threshold).item()

        y_pred_list.extend(pred.argmax(1).cpu().numpy())

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    g_test_loss.append(test_loss)
    correct /= size
    g_test_acc.append(correct)
    if correct > best_acc:
        best_acc = correct
        print("update best acc:", best_acc)
        best_model_name=f"best-model-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pth"
        torch.save(model.state_dict(), best_model_name)
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
        plt.savefig('model.jpg')
        plt.clf()
        plt.close()


if __name__ == '__main__':
    main()