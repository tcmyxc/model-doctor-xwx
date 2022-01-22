# From: https://github.com/kaidic/LDAM-DRW
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    # 不平衡比是100，则 imb_factor= 1/100 = 0.01
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


if __name__ == '__main__':
    import os
    from torchvision import datasets
    imb_factors = [0.01, 0.02, 0.1]
    dataset_names = ["cifar10_lt_ir", "cifar100_lt_ir"]
    for dataset_name in dataset_names:
        for imb_factor in imb_factors:
            if dataset_name == "cifar10_lt_ir":
                trainset = IMBALANCECIFAR10(
                    root='/nfs/xwx/dataset', 
                    train=True, download=True, imb_factor=imb_factor
                )
            elif dataset_name == "cifar100_lt_ir":
                trainset = IMBALANCECIFAR100(
                    root='/nfs/xwx/dataset', 
                    train=True, download=True, imb_factor=imb_factor
                )
            print("train dataset size:", len(trainset))
            trainloader = iter(trainset)
            root_dir = f"/nfs/xwx/dataset/{dataset_name}{int(1/imb_factor)}/images/train"
            print(root_dir)
            # for i, (data, label) in enumerate(trainloader):
            #     # print(i)
            #     # print(data)  # PIL.Image.Image
            #     # data.save("test.png")
            #     img_path = root_dir + "/" + str(label)
            #     if not os.path.exists(img_path):
            #         os.makedirs(img_path)
            #     filename = img_path + '/' + str(i) + '.png'
            #     print(filename)
            #     data.save(filename)

            if dataset_name == "cifar10_lt_ir":
                # 测试集是平衡的
                testset = datasets.CIFAR10(
                    root='/nfs/xwx/dataset', 
                    train=False, download=True
                )
            elif dataset_name == "cifar100_lt_ir":
                # 测试集是平衡的
                testset = datasets.CIFAR100(
                    root='/nfs/xwx/dataset', 
                    train=False, download=True
                )
            print("test dataset size:", len(testset))
            testloader = iter(testset)
            root_dir = f"/nfs/xwx/dataset/{dataset_name}{int(1/imb_factor)}/images/test"
            print(root_dir)
            # for i, (data, label) in enumerate(testloader):
            #     # print(i)
            #     # print(data)  # PIL.Image.Image
            #     # data.save("test.png")
            #     img_path = root_dir + "/" + str(label)
            #     if not os.path.exists(img_path):
            #         os.makedirs(img_path)
            #     filename = img_path + '/' + str(i) + '.png'
            #     print(filename)
            #     data.save(filename)