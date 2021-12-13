import sys

import numpy as np

sys.path.append('/workspace/classification/code/')  # zjl
import os
import torch
import torch.nn.functional as F
import json
import cv2
import matplotlib.pyplot as plt

import models
from configs import config
from utils import image_util


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


def fgsm_attack(model, criterion, inputs, labels, eps, device):
    inputs = inputs.to(device)
    labels = labels.to(device)
    inputs.requires_grad = True

    outputs = model(inputs)

    model.zero_grad()
    loss = criterion(outputs, labels).to(device)
    loss.backward()

    attack_images = inputs + eps * inputs.grad.sign()

    attack_images[:, 0, :, :] = torch.clamp(attack_images[:, 0, :, :], -2.4291, 2.5141)
    attack_images[:, 1, :, :] = torch.clamp(attack_images[:, 1, :, :], -2.4183, 2.5968)
    attack_images[:, 2, :, :] = torch.clamp(attack_images[:, 2, :, :], -2.2214, 2.7537)

    return attack_images


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device('cuda:0')

    data_name = 'cifar-10'
    model_name = 'resnet50'
    model_path = os.path.join(config.model_pretrained, model_name + '_07281512', 'checkpoint.pth')
    data_path = os.path.join(config.output_result, model_name + '_07281512', 'lc')
    cfg = json.load(open('configs/config_trainer.json'))[data_name]
    model = models.load_model(model_name=model_name,
                              in_channels=cfg['model']['in_channels'],
                              num_classes=cfg['model']['num_classes'])
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    model.to(device)

    epsilon = 0.5
    criterion = F.cross_entropy
    data_loader, data_size = load_data(data_path)

    for i, samples in enumerate(data_loader):
        inputs, labels, names = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        h = model(inputs)
        prediction = h.max(1)[1]
        accuracy = torch.eq(prediction, labels).float().mean()
        loss = F.cross_entropy(h, labels)

        inputs_adv = fgsm_attack(model, criterion, inputs, labels, epsilon, device)

        h = model(inputs_adv)
        prediction = h.max(1)[1]
        accuracy_adv = torch.eq(prediction, labels).float().mean()
        loss_adv = F.cross_entropy(h, labels)

        print(accuracy.item(), loss.item(), '|', accuracy_adv.item(), loss_adv.item())

        class_dic = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                     'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}  # cifar
        for i, name in enumerate(names):
            img = image_util.deprocess_image(inputs_adv[i].detach().cpu().numpy(),
                                             mean=(0.4914, 0.4822, 0.4465),
                                             std=(0.2023, 0.1994, 0.2010))
            img_path = os.path.join(config.output_result,
                                    model_name + '_07281512',
                                    'wpas',
                                    list(class_dic.keys())[labels[i]])
            if not os.path.exists(img_path):
                os.makedirs(img_path)

            img = img.transpose(1, 2, 0)
            plt.imsave(os.path.join(img_path, name), img)


if __name__ == '__main__':
    main()

    # from torchvision import transforms
    # from PIL import Image
    # x = np.array([[[0, 0, 0], [255, 255, 255]]])
    #
    # print(x.shape)
    # x_ = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                          (0.2023, 0.1994, 0.2010)),
    # ])(Image.fromarray(x.astype('uint8')).convert('RGB'))
    # print(x_)
    # print(x_.shape)
