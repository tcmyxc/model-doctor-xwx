import sys
sys.path.append('/home/xwx/model-doctor-xwx')

import os
import torch
import json
import models
import pandas as pd
from torchsummary import summary



# config
cfg = json.load(open('configs/config_trainer.json'))["cifar-10"]

model = models.load_model(
    model_name="vgg16",
    in_channels=cfg['model']['in_channels'], # 输入图像的通道数
    num_classes=cfg['model']['num_classes'] # 类别数
)

print(model)
model = model.cuda()
summary(model=model, input_size=(3, 224, 224))
