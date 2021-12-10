import sys

from torch._C import set_anomaly_enabled

sys.path.append('/home/xwx/model-doctor-xwx') #205

import os
import torch
import json

import models


# config
cfg = json.load(open('configs/config_trainer.json'))["cifar-10"]
model_path = os.path.join(
    "/home/xwx/model-doctor-xwx/output/model/gc/resnet50-20211208-101731-final", 
    'checkpoint.pth')

model = models.load_model(model_name="resnet50",
                              in_channels=cfg['model']['in_channels'], # 输入图像的通道数
                              num_classes=cfg['model']['num_classes']) # 类别数

model = model.cuda()
state = torch.load(model_path)['history']
train_loss_cls = state['train_loss_cls']
train_loss_gc = state["train_loss_gc"]
val_acc = state["val_acc"]
best_val_acc = sorted(val_acc, reverse=True)[0]
print(f"best_val_acc is {best_val_acc}")
