import sys

sys.path.append('/workspace/classification/code/')  # zjl
import os
import torch
from torch import nn
import json
import numpy as np

import models
from configs import config


class GradIntegral:
    def __init__(self, model, modules):
        print('- Grad Integral Test')

        self.modules = modules
        self.hooks = []
        self.noise_std = 0

    def add_noise(self, noise_std):
        self.noise_std = noise_std
        for module in self.modules:
            hook = module.register_forward_hook(self._modify_feature_map)
            self.hooks.append(hook)

    def remove_noise(self):
        for hook in self.hooks:
            hook.remove()
            self.hooks.clear()

    def _modify_feature_map(self, module, inputs, outputs):
        noise = torch.normal(mean=0, std=self.noise_std, size=outputs.shape).to(outputs.device)
        outputs += noise


def check_data(model, criterion, data_loader, data_size, device):
    running_loss = 0.0
    running_corrects = 0.0

    for i, samples in enumerate(data_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_acc = running_corrects / data_size
    return epoch_acc


def draw_curves(history, result_path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(np.arange(0, 3, 0.1), history['ns'], 'r', label='ns')
    plt.plot(np.arange(0, 3, 0.1), history['cpas'], 'g', label='cpas')
    plt.plot(np.arange(0, 3, 0.1), history['wpas'], 'b', label='wpas')
    # plt.title("The accuracy curves of normal and adversarial samples with different disturbances to feature")
    plt.xlabel("range of noise")
    plt.ylabel("acc")
    plt.legend(loc="upper right")
    # plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(result_path, 'curves.jpg'))
    plt.clf()


def load_data(data_path):
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from loaders.datasets.image_dataset import ImageDataset

    data_set = ImageDataset(image_dir=data_path,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                     (0.2023, 0.1994, 0.2010)),
                            ]))

    data_loader = DataLoader(dataset=data_set,
                             batch_size=128,
                             num_workers=4,
                             shuffle=True)

    return data_loader, len(data_set)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device('cuda:0')

    data_name = 'cifar-10'
    model_name = 'resnet50'
    model_layers = [-1]
    model_path = os.path.join(config.model_pretrained, model_name + '_07281512', 'checkpoint.pth')
    result_path = os.path.join(config.output_result, model_name + '_07281512')

    cfg = json.load(open('configs/config_trainer.json'))[data_name]

    model = models.load_model(model_name=model_name,
                              in_channels=cfg['model']['in_channels'],
                              num_classes=cfg['model']['num_classes'])

    modules = models.load_modules(model=model,
                                  model_name=model_name,
                                  model_layers=model_layers)

    gi = GradIntegral(model=model, modules=modules)

    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    history = {'ns': [],
               'cpas': [],
               'wpas': []
               }
    for data_type in ['ns', 'cpas', 'wpas']:
        data_path = os.path.join(result_path, data_type)
        data_loader, data_size = load_data(data_path)
        for noise_std in np.arange(0, 3, 0.1):
            gi.remove_noise()
            gi.add_noise(noise_std=noise_std)
            acc = check_data(model, criterion, data_loader, data_size, device)
            history[data_type].append(acc)
            print(data_type, noise_std, acc)

    print(history)
    draw_curves(history, result_path)


if __name__ == '__main__':
    main()
