import os
from torch.utils.data import DataLoader

from configs import config
from torchvision import transforms
from loaders.datasets import image_mask_transforms as im_transforms
from loaders.datasets.image_dataset import ImageDataset
from loaders.datasets.image_mask_dataset import ImageMaskDataset


def load_images(data_type):
    assert data_type in ['train', 'test']

    image_dir = os.path.join(config.data_mnist, data_type)

    if data_type == 'train':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        data_set = ImageDataset(image_dir=image_dir,
                                transform=transform_train)
    else:
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        data_set = ImageDataset(image_dir=image_dir,
                                transform=transform_test)
    data_loader = DataLoader(
        dataset=data_set,
        shuffle=True,
        num_workers=4,
        batch_size=128)

    return data_loader, len(data_set)
