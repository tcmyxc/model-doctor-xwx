"""
生成cifar10数据集
"""
if __name__ == '__main__':
    import os
    from torchvision import datasets
    
    trainset = datasets.CIFAR100(root='/nfs/xwx/dataset', train=True, download=True)
    print("train dataset size:", len(trainset))
    trainloader = iter(trainset)
    root_dir = f"/nfs/xwx/dataset/cifar100/images/train"
    print(root_dir)
    for i, (data, label) in enumerate(trainloader):
        img_path = root_dir + "/" + str(label)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        filename = img_path + '/' + str(i) + '.png'
        # print(filename)
        data.save(filename)
        
    
    testset = datasets.CIFAR100(root='/nfs/xwx/dataset', train=False, download=True)
    print("test dataset size:", len(testset))
    testloader = iter(testset)
    root_dir = f"/nfs/xwx/dataset/cifar100/images/test"
    print(root_dir)
    for i, (data, label) in enumerate(testloader):
        img_path = root_dir + "/" + str(label)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        filename = img_path + '/' + str(i) + '.png'
        # print(filename)
        data.save(filename)
        