import os
from torch.utils.data import DataLoader

from configs import config
from torchvision import transforms
from loaders.datasets import image_mask_transforms as im_transforms
from loaders.datasets.image_dataset import ImageDataset
from loaders.datasets.image_mask_dataset import ImageMaskDataset

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def load_images(data_type):
    assert data_type in ['train', 'test']

    image_dir = os.path.join(config.data_cifar100, data_type)

    if data_type == 'train':
        data_set = ImageDataset(image_dir=image_dir,
                                transform=transforms.Compose([
                                    # transforms.ToPILImage(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor(),
                                    transforms.Normalize(CIFAR100_MEAN,
                                                         CIFAR100_STD)
                                ]))
    else:
        data_set = ImageDataset(image_dir=image_dir,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(CIFAR100_MEAN,
                                                         CIFAR100_STD)
                                ]))
    data_loader = DataLoader(
        dataset=data_set,
        shuffle=True,
        num_workers=4,
        batch_size=128)

    return data_loader, len(data_set)
