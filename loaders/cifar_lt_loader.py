import os, sys
from torch.utils.data import DataLoader
sys.path.append('/nfs/xwx/model-doctor-xwx')
from configs import config
from torchvision import transforms
from loaders.datasets import image_mask_transforms as im_transforms
from loaders.datasets.image_dataset import ImageDataset
from loaders.datasets.image_mask_dataset import ImageMaskDataset

# sampler
from loaders.ClassAwareSampler import get_sampler


def load_cifar_lt_images(data_type, dataset_name):
    assert data_type in ['train', 'test']

    if dataset_name == "cifar-10-lt-ir10":
        image_dir = os.path.join(config.data_cifar10_lt_ir10, data_type)
    elif dataset_name == "cifar-10-lt-ir100":
        image_dir = os.path.join(config.data_cifar10_lt_ir100, data_type)
    elif dataset_name == "cifar-100-lt-ir10":
        image_dir = os.path.join(config.data_cifar100_lt_ir10, data_type)
    elif dataset_name == "cifar-100-lt-ir50":
        image_dir = os.path.join(config.data_cifar100_lt_ir50, data_type)
    elif dataset_name == "cifar-100-lt-ir100":
        image_dir = os.path.join(config.data_cifar100_lt_ir100, data_type)
    
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
                             batch_size=128,
                             num_workers=4,
                             shuffle=True)

    return data_loader, len(data_set)


def load_class_balanced_cifar_lt_images(data_type, dataset_name):
    """类别均衡采样(所有类别都采样相同数量的样本)"""
    
    assert data_type in ['train', 'test']

    if dataset_name == "cifar-10-lt-ir10":
        image_dir = os.path.join(config.data_cifar10_lt_ir10, data_type)
    elif dataset_name == "cifar-10-lt-ir100":
        image_dir = os.path.join(config.data_cifar10_lt_ir100, data_type)
    elif dataset_name == "cifar-100-lt-ir10":
        image_dir = os.path.join(config.data_cifar100_lt_ir10, data_type)
    elif dataset_name == "cifar-100-lt-ir50":
        image_dir = os.path.join(config.data_cifar100_lt_ir50, data_type)
    elif dataset_name == "cifar-100-lt-ir100":
        image_dir = os.path.join(config.data_cifar100_lt_ir100, data_type)
    
    if data_type == 'train':
        data_set = ImageDataset(image_dir=image_dir,
                                transform=transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                         (0.2023, 0.1994, 0.2010)),

                                ]))
        sampler = get_sampler()
        data_loader = DataLoader(dataset=data_set,
                             batch_size=128,
                             num_workers=4,
                             shuffle=False,  # shuffle must be false
                             sampler=sampler(data_set, 4))
    else:
        data_set = ImageDataset(image_dir=image_dir,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                         (0.2023, 0.1994, 0.2010)),
                                ]))
        data_loader = DataLoader(dataset=data_set,
                             batch_size=128,
                             num_workers=4,
                             shuffle=False)

   
    return data_loader, len(data_set)


if __name__ == '__main__':
    from sklearn.metrics import classification_report
    
    dataloader, _ = load_cifar_lt_images("train", "cifar-100-lt-ir100")
    y_pred_list = []
    y_train_list = []
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for batch, (X, y) in enumerate(dataloader):
        y_train_list.extend(y.numpy())
        y_pred_list.extend(y.numpy())
    
    print(classification_report(y_train_list, y_pred_list, digits=4))