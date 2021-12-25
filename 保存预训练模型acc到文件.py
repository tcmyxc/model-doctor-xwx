import sys
sys.path.append('/home/xwx/model-doctor-xwx')

import os
import torch
import json

import models

import pandas as pd

# 训练200轮，每5轮保存一次权重文件，共40个权重文件
# 此脚本的目的在于批量读取40个权重文件中的acc数据，然后写到excel表格，方便后续处理数据

# config
cfg = json.load(open('configs/config_trainer.json'))["cifar-10"]

model = models.load_model(
    model_name="resnet50",
    in_channels=cfg['model']['in_channels'], # 输入图像的通道数
    num_classes=cfg['model']['num_classes'] # 类别数
)

model = model.cuda()

acc_list = []
for epoch in range(5, 201, 5):
# pretrained model path
    model_path = os.path.join(
        "/home/xwx/model-doctor-xwx/output/model/pretrained/",
        "resnet50-cifar-10-prune", 
        f'checkpoint-{epoch}.pth')
    state= torch.load(model_path)
    history = state['best'].history

    val_acc = history["val_acc"]
    best_val_acc = sorted(val_acc, reverse=True)[0]
    acc_list.append((epoch, best_val_acc))
    print(f"epoch {epoch}, val acc is {best_val_acc:.4f}")
    
dataframe = pd.DataFrame(acc_list)
dataframe.to_excel('epoch-prune.xls')