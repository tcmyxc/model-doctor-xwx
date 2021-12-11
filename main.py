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
state= torch.load(model_path)
epoch = state["epoch"] + 1
history = state['history']
train_loss_cls = history['train_loss_cls']
train_loss_gc = history["train_loss_gc"]
val_acc = history["val_acc"]
best_val_acc = sorted(val_acc, reverse=True)[0]
pretrained_acc = 0.9489
print(f"pretrained acc is {pretrained_acc}, best_val_acc is {best_val_acc:.4f}, increase {((best_val_acc-pretrained_acc)*100):.4f}%, epoch is {epoch}")
