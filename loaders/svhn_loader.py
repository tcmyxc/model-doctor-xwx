import sys
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from configs import config


def load_images(data_type):
    assert data_type in ['train', 'test']

    if data_type == 'train':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,))
        ])
        data_set = torchvision.datasets.SVHN(root=config.data_svhn,
                                             split=data_type,
                                             download=True,
                                             transform=transform_train)
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,))
        ])
        data_set = torchvision.datasets.SVHN(root=config.data_svhn,
                                             split=data_type,
                                             download=True,
                                             transform=transform_test)
    data_loader = DataLoader(
        dataset=data_set,
        shuffle=True,
        num_workers=4,
        batch_size=128)

    return data_loader, len(data_set)


if __name__ == '__main__':
    print('==')
    from utils import image_util
    data_loader, data_size = load_images('train')
    print(data_size)
    for i, samples in enumerate(data_loader):
        inputs, targets = samples
        print(inputs.shape)
        image_util.show_torch_images(inputs, mode='gray')
        if i == 1:
            break
