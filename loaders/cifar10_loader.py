import os
from torch.utils.data import DataLoader

from configs import config
from torchvision import transforms
from loaders.datasets import image_mask_transforms as im_transforms
from loaders.datasets.image_dataset import ImageDataset
from loaders.datasets.image_mask_dataset import ImageMaskDataset


def load_images(data_type):
    assert data_type in ['train', 'test']

    image_dir = os.path.join(config.data_cifar10, data_type)

    if data_type == 'train':
        data_set = ImageDataset(image_dir=image_dir,
                                transform=transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                         (0.2023, 0.1994, 0.2010)),

                                ]))

    else:
        data_set = ImageDataset(image_dir=image_dir,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                         (0.2023, 0.1994, 0.2010)),
                                ]))

    data_loader = DataLoader(dataset=data_set,
                             batch_size=64,
                             num_workers=4,
                             shuffle=True)

    return data_loader, len(data_set)


def load_images_masks(data_type):
    assert data_type in ['train', 'test']
    image_dir = os.path.join(config.data_cifar10, data_type)
    mask_dir = os.path.join(config.result_masks_cifar10, data_type)

    if data_type == 'train':
        data_set = ImageMaskDataset(image_dir=image_dir,
                                    mask_dir=mask_dir,
                                    transform=im_transforms.Compose([
                                        im_transforms.RandomCrop(32, padding=4),
                                        im_transforms.RandomHorizontalFlip(),
                                        im_transforms.ToTensor(),
                                        im_transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                (0.2023, 0.1994, 0.2010)),

                                    ]))

    else:
        data_set = ImageMaskDataset(image_dir=image_dir,
                                    mask_dir=mask_dir,
                                    transform=transforms.Compose([
                                        im_transforms.ToTensor(),
                                        im_transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                (0.2023, 0.1994, 0.2010)),
                                    ]))

    data_loader = DataLoader(dataset=data_set,
                             batch_size=128,
                             num_workers=4,
                             shuffle=True)

    return data_loader, len(data_set)


if __name__ == '__main__':
    from utils import image_util

    data_loader, data_size = load_images_masks('train')
    print(data_size)
    img = None
    mask = None
    for i, samples in enumerate(data_loader):
        inputs, labels, masks = samples
        print(inputs.shape)
        print(labels)

        # inputs = inputs / 2 + 0.5  # unnormalize
        image_util.show_torch_images(inputs)
        image_util.show_torch_images(masks)

        mask = masks[0].numpy()
        img = inputs[0].numpy()

        if i == 10:
            break
