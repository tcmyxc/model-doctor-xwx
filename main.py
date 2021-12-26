import sys
sys.path.append('/home/xwx/model-doctor-xwx')

import os
import torch
import json

import models

import pandas as pd


# config
cfg = json.load(open('configs/config_trainer.json'))["cifar-10"]

model = models.load_model(
    model_name="resnet50",
    in_channels=cfg['model']['in_channels'], # 输入图像的通道数
    num_classes=cfg['model']['num_classes'] # 类别数
)

model = model.cuda()

acc_list = []
for epoch in range(0, 201, 5):
    # pretrained model path
    model_path = os.path.join(
        "/home/xwx/model-doctor-xwx/output/model/pretrained/",
        "resnet50-cifar-10-prune", 
        f'checkpoint-{epoch}.pth')
    state= torch.load(model_path)
    val_acc = float(state["acc"])
    acc_list.append((epoch, val_acc))
    print(f"epoch {epoch}, val acc is {val_acc:.4f}")
    
dataframe = pd.DataFrame(acc_list)
dataframe.to_excel('epoch-prune.xls')

