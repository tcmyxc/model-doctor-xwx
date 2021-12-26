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
for epoch in range(0, 10, 5):
    # pretrained+md model path
    model_path = os.path.join(
        "/home/xwx/model-doctor-xwx/output/model",
        "gc",
        "resnet50-cifar-10-prune",
        f"{epoch}",
        'checkpoint.pth')
    if not os.path.exists(model_path):
        print("-" * 79, "\nERROR, the model path does not exist")
        break
    else:
        print("-" * 79, "\nmodel path:", model_path)
    state= torch.load(model_path)
    val_acc = float(state["acc"])
    best_epoch = int(state["epoch"]) + 1  # 保存模型参数时忘记+1了（预训练模型里面的参数没有这个问题）
    acc_list.append((epoch, val_acc))
    print(f"epoch-{epoch} model, val acc is {val_acc:.4f}, best epoch is {best_epoch}")
    
# dataframe = pd.DataFrame(acc_list)
# dataframe.to_excel('epoch-prune.xls')

