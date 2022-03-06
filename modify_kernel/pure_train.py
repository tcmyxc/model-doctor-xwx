# 此脚本只是单纯训练一个分类模型
# 1. 使用普通的数据加载器
# 2. 使用REFL

import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

from loss.refl import reduce_equalized_focal_loss

import time
import torch
from torch import optim
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import models
import loaders
from utils.lr_util import get_lr_scheduler
from trainers.cls_trainer import print_time

# global config
data_name = 'cifar-10-lt-ir100'
model_name = 'resnet32'
epochs = 200
threshold = 0.5
lr = 0.01
momentum = 0.9
weight_decay = 5e-4
in_channels = 3
num_classes = 10


best_acc = 0
g_train_loss, g_train_acc = [], []
g_test_loss, g_test_acc = [], []

def main():
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-' * 79, '\n[Info] train on ', device)

    # data,普通数据加载器
    data_loaders, _ = loaders.load_data(data_name=data_name)

    # model
    model = models.load_model(
        model_name=model_name, 
        in_channels=in_channels,
        num_classes=num_classes
    )
    model.to(device)

    loss_fn = reduce_equalized_focal_loss
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    # lr scheduler
    scheduler = get_lr_scheduler(optimizer, True)

    for epoch in range(epochs):
        epoch_begin_time = time.time()
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        print(f"\nEpoch {epoch+1}")
        print("[INFO] lr is:", cur_lr)
        print("-" * 42)
        train(data_loaders["train"], model, loss_fn, optimizer, device)
        test(data_loaders["val"], model, loss_fn, optimizer, epoch, device)
        scheduler.step()

        draw_acc(g_train_loss, g_test_loss, g_train_acc, g_test_acc)
        print_time(time.time()-epoch_begin_time, epoch=True)
        
    print("Done!")


def train(dataloader, model, loss_fn, optimizer, device):
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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    correct /= size
    g_train_loss.append(train_loss)
    g_train_acc.append(correct)
    print("-" * 42)
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
        print(f"[FEAT] Epoch {epoch+1}, update best acc:", best_acc)
        best_model_name = f"{model_name}_{data_name}_best_model.pth"
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
        plt.savefig('pure_train_cls_model.jpg')
        plt.clf()
        plt.close()


if __name__ == '__main__':
    main()