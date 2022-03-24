import numpy as np
import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

import torch
import numpy as np
import os
import json

import models
from configs import config
from utils import image_util
from utils import data_util

from mylog import log

train_log = log()

result_path = os.path.join("/nfs/xwx/model-doctor-xwx/output/result/resnet32-cifar-10-lt-ir100-all", 'grads')

data_name = 'cifar-10-lt-ir100'
model_name = 'resnet32'

# config
cfg = json.load(open('configs/config_trainer.json'))[data_name]
device = torch.device('cuda:0')

# model
model = models.load_model(model_name=model_name,
                            in_channels=cfg['model']['in_channels'],
                            num_classes=cfg['model']['num_classes'])

# modules
modules = models.load_modules(model=model, model_name=model_name, model_layers=None)  # no first conv

for idx in range(cfg['model']['num_classes']):
    kernel_dict = {}

    for layer in range(len(modules)):
        for label in range(cfg['model']['num_classes']):
            mask_root_path = os.path.join(result_path, str(layer), str(label))
            method_name = 'inputs_label{}_layer{}'.format(label, layer)
            mask_path = os.path.join(mask_root_path, 'grads_{}.npy'.format(method_name))
            if label == idx:
                data = np.load(mask_path)
                # print(f"layer {layer}, label {label}", np.where(data==1))
                kernel_num = data.size
                kernel_valid = np.where(np.isin(data, 1))[0].tolist()
                kernel_val = []
                kernel_val.append(kernel_num)
                kernel_val.append(kernel_valid)
                kernel_dict[layer] = kernel_val

    np.save(f"kernel_dict_label_{idx}.npy", kernel_dict)
    print(kernel_dict)
    print("-"*40)