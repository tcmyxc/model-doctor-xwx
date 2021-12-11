import os
from torch.utils.data import DataLoader

from configs import config
from torchvision import transforms
from loaders.datasets import image_mask_transforms as im_transforms
from loaders.datasets.image_dataset import ImageDataset
from loaders.datasets.image_mask_dataset import ImageMaskDataset

from utils import image_util


def load_images(data_type):
    assert data_type in ['train', 'test']

    image_dir = os.path.join(config.data_mini_imagenet, data_type)

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
                             batch_size=32,
                             num_workers=4,
                             shuffle=True)

    return data_loader, len(data_set)


def load_images_masks(data_type):
    assert data_type in ['train']
    image_dir = os.path.join(config.data_mini_imagenet, data_type)
    mask_dir = config.result_masks_mnim

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


if __name__ == '__main__':
    data_loader, data_size = load_images_masks('train')
    print(data_size)
    img = None
    mask = None
    for i, samples in enumerate(data_loader):
        inputs, targets, masks = samples
        print(inputs.shape)
        print(masks.shape)
        print(targets)

        masks = transforms.Resize((56, 56))(masks)
        print(masks[0])

        # inputs = inputs / 2 + 0.5  # unnormalize
        image_util.show_torch_images(masks)
        image_util.show_torch_images(inputs)

        mask = masks[0].numpy()
        img = inputs[0].numpy()

        if i == 0:
            break
