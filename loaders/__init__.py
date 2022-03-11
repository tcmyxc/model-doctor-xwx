from loaders.cifar10_loader import load_images as load_cifar10_images
from loaders.cifar10_loader import load_images_masks as load_cifar10_images_masks
from loaders.cifar100_loader import load_images as load_cifar100_images
from loaders.mnist_loader import load_images as load_mnist_images
from loaders.fashion_mnist_loader import load_images as load_fashion_mnist_images
from loaders.stl10_loader import load_images as load_stl10_images
from loaders.stl10_loader import load_images_masks as load_stl10_images_masks
from loaders.mnin_loader import load_images as load_mnin_images
from loaders.mnin_loader import load_images_masks as load_mnin_images_masks

# for long-tailed dataset
from loaders.imagenet_lt_loader import load_images as load_imagenet_lt_images
from loaders.imagenet_lt_loader import load_class_balanced_imagenet_lt_images

from loaders.imagenet_10_lt_loader import load_images as load_imagenet_10_lt_images
from loaders.imagenet_10_lt_loader import load_class_balanced_imagenet_10_lt_images

from loaders.cifar_lt_loader import load_cifar_lt_images
from loaders.cifar_lt_loader import load_class_balanced_cifar_lt_images

from loaders.inaturalist_data_loaders import load_images as load_inaturalist2018_images


def load_data(data_name, data_type=None, with_mask=False):
    print('-' * 40)
    print('LOAD DATA:', data_name)
    print('-' * 40)

    if data_type is None:
        train_loader, test_loader, train_size, test_size = None, None, None, None
        if data_name == 'cifar-10':
            train_loader, train_size = load_cifar10_images('train')
            test_loader, test_size = load_cifar10_images('test')
        elif data_name == 'imagenet-lt':
            train_loader, train_size = load_imagenet_lt_images('train')
            test_loader, test_size = load_imagenet_lt_images('test')
        elif data_name == 'imagenet-10-lt':
            train_loader, train_size = load_imagenet_10_lt_images('train')
            test_loader, test_size = load_imagenet_10_lt_images('test')
        elif data_name == "inaturalist2018":
            train_loader, train_size = load_inaturalist2018_images("train")
            test_loader, test_size = load_inaturalist2018_images("test")
        elif data_name == 'cifar-10-lt-ir10':
            train_loader, train_size = load_cifar_lt_images("train", data_name)
            test_loader, test_size = load_cifar_lt_images("test", data_name)
        elif data_name == 'cifar-10-lt-ir50':
            train_loader, train_size = load_cifar_lt_images("train", data_name)
            test_loader, test_size = load_cifar_lt_images("test", data_name)
        elif data_name == 'cifar-10-lt-ir100':
            train_loader, train_size = load_cifar_lt_images("train", data_name)
            test_loader, test_size = load_cifar_lt_images("test", data_name)
        elif data_name == 'cifar-100':
            train_loader, train_size = load_cifar100_images('train')
            test_loader, test_size = load_cifar100_images('test')
        elif data_name == 'cifar-100-lt-ir10':
            train_loader, train_size = load_cifar_lt_images("train", data_name)
            test_loader, test_size = load_cifar_lt_images("test", data_name)
        elif data_name == 'cifar-100-lt-ir50':
            train_loader, train_size = load_cifar_lt_images("train", data_name)
            test_loader, test_size = load_cifar_lt_images("test", data_name)
        elif data_name == 'cifar-100-lt-ir100':
            train_loader, train_size = load_cifar_lt_images("train", data_name)
            test_loader, test_size = load_cifar_lt_images("test", data_name)
        elif data_name == 'mnist':
            train_loader, train_size = load_mnist_images('train')
            test_loader, test_size = load_mnist_images('test')
        elif data_name == 'fashion-mnist':
            train_loader, train_size = load_fashion_mnist_images('train')
            test_loader, test_size = load_fashion_mnist_images('test')
        elif data_name == 'stl-10':
            if with_mask:
                train_loader, train_size = load_stl10_images_masks('train')
                test_loader, test_size = load_stl10_images('test')
            else:
                train_loader, train_size = load_stl10_images('train')
                test_loader, test_size = load_stl10_images('test')
        elif data_name == 'mini-imagenet':
            if with_mask:
                train_loader, train_size = load_mnin_images_masks('train')
                test_loader, test_size = load_mnin_images('test')
            else:
                train_loader, train_size = load_mnin_images('train')
                test_loader, test_size = load_mnin_images('test')
        elif data_name == 'mini-imagenet-10':
            if with_mask:
                train_loader, train_size = load_mnin_images_masks('train')
                test_loader, test_size = load_mnin_images('test')
            else:
                train_loader, train_size = load_mnin_images('train')
                test_loader, test_size = load_mnin_images('test')
        data_loaders = {'train': train_loader, 'val': test_loader}
        dataset_sizes = {'train': train_size, 'val': test_size}
        return data_loaders, dataset_sizes
    else:
        if data_name == 'cifar-10':
            return load_cifar10_images(data_type)
        elif data_name == 'imagenet-lt':
            return load_imagenet_lt_images(data_type)
        elif data_name == 'imagenet-10-lt':
            return load_imagenet_10_lt_images(data_type)
        elif data_name == "inaturalist2018":
            return load_inaturalist2018_images(data_type)
        elif data_name == 'cifar-10-lt-ir10':
            return load_cifar_lt_images(data_type, data_name)
        elif data_name == 'cifar-10-lt-ir50':
            return load_cifar_lt_images(data_type, data_name)
        elif data_name == 'cifar-10-lt-ir100':
            return load_cifar_lt_images(data_type, data_name)
        elif data_name == 'cifar-100':
            return load_cifar100_images(data_type)
        elif data_name == 'cifar-100-lt-ir10':
            return load_cifar_lt_images(data_type, data_name)
        elif data_name == 'cifar-100-lt-ir50':
            return load_cifar_lt_images(data_type, data_name)
        elif data_name == 'cifar-100-lt-ir100':
            return load_cifar_lt_images(data_type, data_name)
        elif data_name == 'mnist':
            return load_mnist_images(data_type)
        elif data_name == 'fashion-mnist':
            return load_fashion_mnist_images(data_type)
        elif data_name == 'stl-10':
            if with_mask:
                return load_stl10_images_masks(data_type)
            else:
                return load_stl10_images(data_type)
        elif data_name == 'mini-imagenet':
            if with_mask:
                return load_mnin_images_masks(data_type)
            else:
                return load_mnin_images(data_type)
        elif data_name == 'mini-imagenet-10':
            if with_mask:
                return load_mnin_images_masks(data_type)
            else:
                return load_mnin_images(data_type)


# 通过类平衡采样加载数据
def load_class_balanced_data(data_name, data_type=None):
    print('Load class balanced data:', data_name)

    if data_type is None:
        train_loader, test_loader, train_size, test_size = None, None, None, None
        if data_name == 'cifar-10-lt-ir10':
            train_loader, train_size = load_class_balanced_cifar_lt_images("train", data_name)
            test_loader, test_size = load_class_balanced_cifar_lt_images("test", data_name)
        elif data_name == 'cifar-10-lt-ir50':
            train_loader, train_size = load_class_balanced_cifar_lt_images("train", data_name)
            test_loader, test_size = load_class_balanced_cifar_lt_images("test", data_name)
        elif data_name == 'cifar-10-lt-ir100':
            train_loader, train_size = load_class_balanced_cifar_lt_images("train", data_name)
            test_loader, test_size = load_class_balanced_cifar_lt_images("test", data_name)
        elif data_name == 'cifar-100-lt-ir10':
            train_loader, train_size = load_class_balanced_cifar_lt_images("train", data_name)
            test_loader, test_size = load_class_balanced_cifar_lt_images("test", data_name)
        elif data_name == 'cifar-100-lt-ir50':
            train_loader, train_size = load_class_balanced_cifar_lt_images("train", data_name)
            test_loader, test_size = load_class_balanced_cifar_lt_images("test", data_name)
        elif data_name == 'cifar-100-lt-ir100':
            train_loader, train_size = load_class_balanced_cifar_lt_images("train", data_name)
            test_loader, test_size = load_class_balanced_cifar_lt_images("test", data_name)
        elif data_name == "imagenet-lt":
            train_loader, train_size = load_class_balanced_imagenet_lt_images("train")
            test_loader, test_size   = load_class_balanced_imagenet_lt_images("test")
        elif data_name == "imagenet-10-lt":
            train_loader, train_size = load_class_balanced_imagenet_10_lt_images("train")
            test_loader, test_size   = load_class_balanced_imagenet_10_lt_images("test")
       
        data_loaders = {'train': train_loader, 'val': test_loader}
        dataset_sizes = {'train': train_size, 'val': test_size}
        return data_loaders, dataset_sizes
    else:
        if data_name == 'cifar-10-lt-ir10':
            return load_class_balanced_cifar_lt_images(data_type, data_name)
        elif data_name == 'cifar-10-lt-ir50':
            return load_class_balanced_cifar_lt_images(data_type, data_name)
        elif data_name == 'cifar-10-lt-ir100':
            return load_class_balanced_cifar_lt_images(data_type, data_name)
        elif data_name == 'cifar-100-lt-ir10':
            return load_class_balanced_cifar_lt_images(data_type, data_name)
        elif data_name == 'cifar-100-lt-ir50':
            return load_class_balanced_cifar_lt_images(data_type, data_name)
        elif data_name == 'cifar-100-lt-ir100':
            return load_class_balanced_cifar_lt_images(data_type, data_name)
        elif data_name == "imagenet-lt":
            return load_class_balanced_imagenet_lt_images(data_type)
        elif data_name == "imagenet-10-lt":
            return load_class_balanced_imagenet_10_lt_images(data_type)
