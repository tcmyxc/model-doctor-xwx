import os
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop
from torch.utils.data import DataLoader
from torchvision import transforms
from loaders.datasets.image_dataset import ImageDataset


def load_data(data_path, data_name=None):
    """加载高置信度图片的 dataloader"""
    # 后期需要根据数据集名称添加不同的 transform
    if "cifar" in data_name:
        train_trsfm = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.2023, 0.1994, 0.2010)),
                                ])
    elif "imagenet" in data_name:
        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    data_set = ImageDataset(image_dir=data_path, transform=train_trsfm)
    data_loader = DataLoader(dataset=data_set,
                                batch_size=4,
                                num_workers=4,
                                shuffle=True)
    return data_loader


def load_single_data(data_path):
    # img, tensor
    img = cv2.imread(data_path, 1)[:, :, ::-1]
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    data_tensor = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225])
    ])(img.copy()).unsqueeze(0)

    # label
    img_root_path, class_name = os.path.split(os.path.split(data_path)[0])
    class_names = sorted([d.name for d in os.scandir(img_root_path) if d.is_dir()])
    class_indices = {class_names[i]: i for i in range(len(class_names))}
    label = torch.tensor(class_indices[class_name]).unsqueeze(0)

    return data_tensor, label, img
