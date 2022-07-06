# From: https://github.com/kaidic/LDAM-DRW
import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
# sampler
from loaders.ClassAwareSampler import get_sampler
from configs import config
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, reverse=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse)
        self.gen_imbalanced_data(img_num_list)
        self.reverse = reverse

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num =  img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))                    
                else:
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        return img, target, target

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


def load_cifar_lt_images(data_type, dataset_name):
    assert data_type in ['train', 'test']
    image_dir = config.output_data
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    train_transforms = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                    ])
    test_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize,
                                ])

    if dataset_name == "cifar-10-lt-ir10":
        if data_type == 'train':
            data_set = IMBALANCECIFAR10(root=image_dir, imb_factor=0.1, train=True, transform=train_transforms)
        else:
            data_set = datasets.CIFAR10(root=image_dir, train=False, transform=test_transforms)
    elif dataset_name == "cifar-10-lt-ir100":
        if data_type == 'train':
            data_set = IMBALANCECIFAR10(root=image_dir, imb_factor=0.01, train=True, transform=train_transforms)
        else:
            data_set = datasets.CIFAR10(root=image_dir, train=False, transform=test_transforms)
    elif dataset_name == "cifar-100-lt-ir10":
        if data_type == 'train':
            data_set = IMBALANCECIFAR100(root=image_dir, imb_factor=0.1, train=True, transform=train_transforms)
        else:
            data_set = datasets.CIFAR100(root=image_dir, train=False, transform=test_transforms)
    elif dataset_name == "cifar-100-lt-ir50":
        if data_type == 'train':
            data_set = IMBALANCECIFAR100(root=image_dir, imb_factor=0.02, train=True, transform=train_transforms)
        else:
            data_set = datasets.CIFAR100(root=image_dir, train=False, transform=test_transforms)
    elif dataset_name == "cifar-100-lt-ir100":
        if data_type == 'train':
            data_set = IMBALANCECIFAR100(root=image_dir, imb_factor=0.01, train=True, transform=train_transforms)
        else:
            data_set = datasets.CIFAR100(root=image_dir, train=False, transform=test_transforms)

    data_loader = DataLoader(dataset=data_set,
                             batch_size=128,
                             num_workers=4,
                             shuffle=True)

    return data_loader, len(data_set)


def load_class_balanced_cifar_lt_images(data_type, dataset_name):
    """类别均衡采样(所有类别都采样相同数量的样本)"""
    
    assert data_type in ['train', 'test']

    image_dir = config.output_data
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    train_transforms = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                    ])
    test_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize,
                                ])

    if dataset_name == "cifar-10-lt-ir10":
        if data_type == 'train':
            data_set = IMBALANCECIFAR10(root=image_dir, imb_factor=0.1, train=True, transform=train_transforms)
        else:
            data_set = datasets.CIFAR10(root=image_dir, train=False, transform=test_transforms)
    elif dataset_name == "cifar-10-lt-ir100":
        if data_type == 'train':
            data_set = IMBALANCECIFAR10(root=image_dir, imb_factor=0.01, train=True, transform=train_transforms)
        else:
            data_set = datasets.CIFAR10(root=image_dir, train=False, transform=test_transforms)
    elif dataset_name == "cifar-100-lt-ir10":
        if data_type == 'train':
            data_set = IMBALANCECIFAR100(root=image_dir, imb_factor=0.1, train=True, transform=train_transforms)
        else:
            data_set = datasets.CIFAR100(root=image_dir, train=False, transform=test_transforms)
    elif dataset_name == "cifar-100-lt-ir50":
        if data_type == 'train':
            data_set = IMBALANCECIFAR100(root=image_dir, imb_factor=0.02, train=True, transform=train_transforms)
        else:
            data_set = datasets.CIFAR100(root=image_dir, train=False, transform=test_transforms)
    elif dataset_name == "cifar-100-lt-ir100":
        if data_type == 'train':
            data_set = IMBALANCECIFAR100(root=image_dir, imb_factor=0.01, train=True, transform=train_transforms)
        else:
            data_set = datasets.CIFAR100(root=image_dir, train=False, transform=test_transforms)
    
    if data_type == 'train':
        sampler = get_sampler()
        data_loader = DataLoader(dataset=data_set,
                             batch_size=128,
                             num_workers=4,
                             shuffle=False,  # shuffle must be false
                             sampler=sampler(data_set, 4))
    else:
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

    for batch, (X, y, _) in enumerate(dataloader):
        y_train_list.extend(y.numpy())
        y_pred_list.extend(y.numpy())
    
    print(classification_report(y_train_list, y_pred_list, digits=4))