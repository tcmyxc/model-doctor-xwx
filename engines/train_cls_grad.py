import sys
import os
import argparse
import torch
from torch import optim
from torch import nn
import json
import logging
import datetime

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
    print("-" * 79, "\n[Info]: TRAIN ON DEVICE:", device)
    # logging.info(f"\nTRAIN ON DEVICE: {device}")
    # data
    data_loaders, dataset_sizes = loaders.load_data(data_name=args.data_name, with_mask=False)
    # model
    model = models.load_model(
        model_name=args.model_name,
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"]
    )
    # modules
    modules = models.load_modules(
        model=model,
        model_name=args.model_name,
        model_layers=args.model_layers
    )

    criterion = nn.CrossEntropyLoss()
    # ----------------------------------------
    # train
    # ----------------------------------------
    for epoch in range(0, 1):
        # 2021-12-27，修改学习率没有在训练开始的时候重置的问题
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
        # 检查学习率
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        print("\n==> lr:", cur_lr)

        # 2021-12-25 modify
        # pretrained model path
        model_path = os.path.join(
            config.model_pretrained, 
            args.pretrained_name, 
            'checkpoint.pth'
        )
        if not os.path.exists(model_path):
            print("\n[ERROR]: pretrained model_path does not exist")
            return
        else:
            print("\n[Info]: pretrained model path:", model_path)
        
        # channel path
        channel_paths = [os.path.join(config.result_channels,
                                    args.pretrained_name,
                                    'channels_{}_epoch{}.npy'.format(layer, epoch)) for layer in args.model_layers]
        if not os.path.exists(channel_paths[0]):
            print("\n[ERROR]: result_channels path does not exist")
            return
        else:
            print("\n[Info]: channel_paths:", channel_paths)

        # result path
        result_path = os.path.join(config.output_model, args.result_name, str(epoch))
        print("\n[Info]: result_path:", result_path)
        # return
        
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

def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

if __name__ == '__main__':
    

    logging.Formatter.converter = beijing

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO", # "DEBUG" "WARNING" "ERROR"
        stream=sys.stdout,
    )

    main()
