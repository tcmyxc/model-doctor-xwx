import os
from torch.utils.data import DataLoader

from configs import config
from torchvision import transforms
from loaders.datasets import image_mask_transforms as im_transforms
from loaders.datasets.image_dataset import ImageDataset
from loaders.datasets.image_mask_dataset import ImageMaskDataset


def load_images(data_type):
    assert data_type in ['train', 'test']

    image_dir = os.path.join(config.data_stl10, data_type)

    if data_type == 'train':
        data_set = ImageDataset(image_dir=image_dir,
                                transform=transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.RandomVerticalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])

                                ]))

    else:
        data_set = ImageDataset(image_dir=image_dir,
                                transform=transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
                                ]))

    data_loader = DataLoader(dataset=data_set,
                             batch_size=16,
                             num_workers=4,
                             shuffle=True)

    return data_loader, len(data_set)


def load_images_masks(data_type):
    assert data_type in ['train']
    image_dir = os.path.join(config.data_stl10, data_type)
    mask_dir = config.result_masks_stl10

    if data_type == 'train':
        data_set = ImageMaskDataset(image_dir=image_dir,
                                    mask_dir=mask_dir,
                                    transform=im_transforms.Compose([
                                        im_transforms.Resize((224, 224)),
                                        im_transforms.RandomHorizontalFlip(0.5),
                                        im_transforms.RandomVerticalFlip(0.5),
                                        im_transforms.ToTensor(),
                                        im_transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])

                                    ]))

        # data loader
        data_loader = DataLoader(dataset=data_set,
                                 batch_size=32,
                                 num_workers=4,
                                 shuffle=True)

        return data_loader, len(data_set)
