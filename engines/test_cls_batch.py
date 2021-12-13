import sys

sys.path.append('/workspace/classification/code/')  # zjl
# sys.path.append('/nfs3-p1/hjc/classification/code/')  # vipa

import os
import torch
from torch import nn
import json

import models
import loaders
from configs import config


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    phase = 'val'

    # ----------------------------------------
    # initial
    # ----------------------------------------
    cfg = json.load(open('configs/config_trainer.json'))[data_name]
    data_loaders, dataset_sizes = loaders.load_data(data_name=data_name)
    model = models.load_model(model_name=model_name,
                              in_channels=cfg['model']['in_channels'],
                              num_classes=cfg['model']['num_classes'])
    print(model_path)
    state = torch.load(model_path + '/checkpoint.pth')
    model.load_state_dict(state['model'])
    print(state['epoch'])
    # model.load_state_dict(torch.load(model_path + '/checkpoint.pth')['model'])
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------
    # test
    # ----------------------------------------
    running_loss = 0.0
    running_corrects = 0.0

    for i, samples in enumerate(data_loaders[phase]):
        if i % 10 == 0:
            print('\r{}/{}'.format(i, len(data_loaders[phase])), end='', flush=True)

        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects / dataset_sizes[phase]
    print('\r' + model_path)
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


if __name__ == '__main__':
    # print('-' * 40)
    # print('Check Data Type:', phase)
    # print('Load Model From:', model_path)
    # print('Test On Device:', device)
    # print('-' * 40)

    data_name = 'mini-imagenet'
    model_list = [
        'alexnet',
        'vgg16',
        'resnet50',
        'senet34',
        # 'wideresnet28',
        # 'resnext50',
        'densenet121',
        'simplenetv1',
        'efficientnetv2s',
        'googlenet',
        'xception',
        'mobilenetv2',
        # 'inceptionv3',
        'shufflenetv2',
        'squeezenet',
        'mnasnet'
    ]
    for model_name in model_list:
        result_name = 'pretrained/' + model_name + '_09091631'
        model_path = os.path.join(config.output_model, result_name)
        main()
