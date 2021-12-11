import sys
import os
import argparse
import torch
from torch import optim
from torch import nn
import json
import logging

import models
import loaders
from configs import config
from trainers.cls_grad_trainer import ClsGradTrainer


def main():
    parser = argparse.ArgumentParser(description='CLASSIFICATION MODEL TRAINER')
    parser.add_argument('--data_name', default='cifar-10', type=str, help='data name')
    parser.add_argument('--model_name', default='resnet50', type=str, help='model name')
    parser.add_argument('--result_name', default='gc/resnet50', type=str, help='result name')
    parser.add_argument('--pretrained_name', default='resnet50', type=str, help='pretrained name')
    parser.add_argument('--model_layers', default='-1', nargs='+', type=int, help='model layers')
    parser.add_argument('--device_index', default='0', type=str, help='device index')
    args = parser.parse_args()

    # config
    cfg = json.load(open('configs/config_trainer.json'))[args.data_name]
    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('TRAIN ON DEVICE:', device)
    # data
    data_loaders, dataset_sizes = loaders.load_data(data_name=args.data_name, with_mask=True)
    # model
    model = models.load_model(
        model_name=args.model_name,
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"]
    )
    modules = models.load_modules(
        model=model,
        model_name=args.model_name,
        model_layers=args.model_layers
    )

    # train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=cfg['optimizer']['lr'],
        momentum=cfg['optimizer']['momentum'],
        weight_decay=cfg['optimizer']['weight_decay']
    )
    # 使用余弦退火方案设置每个参数组的学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cfg['scheduler']['T_max']
    )

    # ----------------------------------------
    # train
    # ----------------------------------------
    result_path = os.path.join(config.output_model, args.result_name)
    print('=' * 42)
    print('CHECK RESULT PATH:', result_path)

    model_path = os.path.join(config.model_pretrained, args.pretrained_name, 'checkpoint.pth')
    if not os.path.exists(model_path):
        print('=' * 42)
        print("pretrained model_path 路径不存在")
        return

    if not os.path.exists(config.result_channels):
        print("result_channels 路径不存在")
        return
    channel_paths = [os.path.join(config.result_channels,
                                  args.pretrained_name,
                                  'channels_{}.npy'.format(layer)) for layer in args.model_layers]

    trainer = ClsGradTrainer(
        model=model,
        modules=modules,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        data_loaders=data_loaders,
        dataset_sizes=dataset_sizes,
        num_classes=cfg['model']['num_classes'],
        num_epochs=cfg['trainer']['num_epochs'],
        result_path=result_path,
        model_path=model_path,
        channel_paths=channel_paths
    )

    trainer.train()
    # trainer.check()


if __name__ == '__main__':
    main()
