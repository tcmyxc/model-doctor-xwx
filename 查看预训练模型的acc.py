import sys
sys.path.append('/home/xwx/model-doctor-xwx')

import os
import torch
import json

import models

import pandas as pd


# 数据集
cfg = json.load(open('configs/config_trainer.json'))["stl-10"]

# 模型
model = models.load_model(
    model_name="alexnet",
    in_channels=cfg['model']['in_channels'], # 输入图像的通道数
    num_classes=cfg['model']['num_classes'] # 类别数
)

model = model.cuda()

acc_list = []

# 模型路径
model_path = os.path.join(
    "/home/xwx/model-doctor-xwx/output/model",
    "gc",
    "alexnet-stl-10-md-2",
    'checkpoint.pth')
if not os.path.exists(model_path):
    print("-" * 79, f"\n ERROR, the model path does not exist")
else:
    print("-" * 79, "\n model path:", model_path)
    state= torch.load(model_path)
    epoch = state["epoch"] + 1
    history = state['history']
    train_loss_cls = history['train_loss_cls']
    train_loss_gc = history["train_loss_gc"]
    val_acc = history["val_acc"]
    best_val_acc = sorted(val_acc, reverse=True)[0]
    print(f"current val acc is {best_val_acc:.4f}")


# dataframe = pd.DataFrame(acc_list, columns=["epoch", "acc"])
# dataframe.to_excel('epoch-prune-md.xls')

