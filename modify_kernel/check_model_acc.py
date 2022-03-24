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


threshold = 0.5

def main():
    # cfg
    data_name = 'cifar-100-lt-ir100'
    model_name = 'resnet32'

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-' * 79, '\n[Info] train on ', device)

    # data
    data_loaders, _ = loaders.load_data(data_name=data_name)

    # model
    model = models.load_model(
        model_name=model_name,
        in_channels=3,
        num_classes=100
    )
    
    cp_path = os.path.join('/nfs/xwx/model-doctor-xwx/best-model-20220309-150622-acc0.2323.pth')
    if not os.path.exists(cp_path):
        print("=" * 40)
        print("模型文件的路径不存在, 请检查")
        return
    state = torch.load(cp_path)["model"]
   
    model.load_state_dict(state)
    model.to(device)


    # cfg
    loss_fn = reduce_equalized_focal_loss

    model.eval()
    test(data_loaders["val"], model, loss_fn, device)


def test(dataloader, model, loss_fn, device):
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
    correct /= size
    print(f"Test Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    print(classification_report(y_train_list, y_pred_list, digits=4))


if __name__ == '__main__':
    main()