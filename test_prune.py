import sys
import os
import argparse
import torch
from torch import optim
from torch import nn
import json
import logging
import torch.nn.utils.prune as prune
import torch.nn.functional as F

import models
import loaders
from configs import config
from trainers.cls_grad_trainer import ClsGradTrainer

def main():
    cfg = json.load(open('configs/config_trainer.json'))["cifar-10"]
    model = models.load_model(
        model_name="resnet50",
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"]
    )

    # with open("resnet50_arch.txt", "w") as f:
    #     print(model, file=f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-" * 79)
    print("device:", device)

    model = model.to(device)

    model_layers = None
    modules = models.load_modules(model=model, model_name="resnet50",model_layers=model_layers)
    for i in enumerate(modules):
        print(i)

    # module = model.conv1[0]
    # print("-" * 79)
    # for name, parameters in module.named_parameters():
    #     print(name, ':', parameters.size())
    #
    # print("-" * 79)
    # print(model.state_dict().keys())
    #
    # # for name, parameters in model.named_parameters():
    # #     print(name, ':', parameters.size())
    #
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         pass
    #
    # print("-" * 79)
    # print(dict(model.named_buffers()).keys())


if __name__ == '__main__':
    main()