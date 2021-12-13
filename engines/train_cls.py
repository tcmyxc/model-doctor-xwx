import sys
import os
import argparse
import json

import torch
from torch import optim
from torch import nn

import models
import loaders
from configs import config
from trainers.cls_trainer import ClsTrainer


def main():
    parser = argparse.ArgumentParser(description='CLASSIFICATION MODEL TEST')
    parser.add_argument('--data_name', default='cifar-10', type=str, help='data name')
    parser.add_argument('--model_name', default='resnet50', type=str, help='model name')
    parser.add_argument('--result_name', default='resnet50', type=str, help='save name')
    parser.add_argument('--device_index', default='0', type=str, help='device index')
    args = parser.parse_args()

    # init
    # config
    cfg = json.load(open('configs/config_trainer.json'))[args.data_name]
    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('TRAIN ON DEVICE:', device)
    # data
    data_loaders, dataset_sizes = loaders.load_data(data_name=args.data_name)
    # model
    model = models.load_model(
        model_name=args.model_name,
        in_channels=cfg['model']['in_channels'],
        num_classes=cfg['model']['num_classes']
    )
    # train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=cfg['optimizer']['lr'],
        momentum=cfg['optimizer']['momentum'],
        weight_decay=cfg['optimizer']['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cfg['scheduler']['T_max']
    )

    # ----------------------------------------
    # train
    # ----------------------------------------
    result_path = os.path.join(config.output_model, args.result_name)
    print('-' * 40)
    print('CHECK RESULT PATH:', result_path)
    print('-' * 40)

    trainer = ClsTrainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        data_loaders=data_loaders,
        dataset_sizes=dataset_sizes,
        num_classes=cfg['model']['num_classes'],
        num_epochs=cfg['trainer']['num_epochs'],
        result_path=result_path,
        model_path=None
    )

    trainer.train()
    # trainer.check()


if __name__ == '__main__':
    main()
