# 功能概述
# 1. 使用stage2训练好的模型挑选高置信度图片
# 2. 使用高置信度图片找类别有关的卷积核
# 3. 训练模型

import sys
sys.path.append('/nfs/xwx/model-doctor-xwx') #205

import os
import torch
import argparse
import matplotlib
import yaml
import datetime
import time
import loaders
import models

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from torch import optim
from configs import config
from utils import data_util, image_util
from core.image_sift import ImageSift
from core.pattern_sift import GradSift
from utils.lr_util import get_lr_scheduler
from trainers.cls_trainer import print_time
from loss.refl import reduce_equalized_focal_loss
from sklearn.metrics import classification_report


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='imagenet-10-lt')
parser.add_argument('--threshold', type=float, default='0.5')


# todo：再写一个函数，只是普通加载数据，不使用类别平衡采样（已写好cifar）


modify_dicts = []
best_acc = 0
g_train_loss, g_train_acc = [], []
g_test_loss, g_test_acc = [], []


def main():
    args = parser.parse_args()
    print(f"\n[INFO] args: {args} \n")

    threshold = args.threshold
    data_name = args.data_name

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')
    cfg_filename = "cbs_refl.yml"
    cfg = get_cfg(cfg_filename)[data_name]

    print("-" * 42)
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("-" * 42)

    dataset_root = get_dataset_root(data_name)
    check_path(dataset_root)
    sift_image_path = get_sift_image(cfg, dataset_root, device, args)
    check_path(sift_image_path)
    # sift_image_path = "/nfs/xwx/model-doctor-xwx/output/result/resnet32-cifar-100-lt-ir100/stage3/high/images"
    grad_result_path = find_kernel(cfg, sift_image_path, device, args)
    # grad_result_path = "/nfs/xwx/model-doctor-xwx/output/result/resnet32-cifar-100-lt-ir100/stage3/grads"
    view_layer_kernel(grad_result_path, cfg, args)
    check_path(grad_result_path)
    kernel_dict_root_path = union_cls_kernel(cfg, grad_result_path, args)
    check_path(kernel_dict_root_path)
    # kernel_dict_root_path = "/nfs/xwx/model-doctor-xwx/modify_kernel/kernel_dict/resnet32-cifar-10-lt-ir100"
    # train_and_val(cfg, kernel_dict_root_path, device)


def get_dataset_root(data_name):
    # todo: 其他数据集
    if data_name == "imagenet-lt":
        return config.data_imagenet_lt
    elif data_name == "imagenet-10-lt":
        return config.data_imagenet_lt
    elif data_name == "cifar-100-lt-ir100":
        return config.data_cifar100_lt_ir100
    

def check_path(path, msg=None):
    """检查路径是否合法"""
    if not os.path.exists(path):
        if msg == None:
            print("\n[ERROR] path does not exist")
        else:
            print(f"\n[ERROR] {msg} does not exist")
        sys.exit(1)
    else:
        if msg == None:
            print("\n[INFO] path:", path)
        else:
            print(f"\n[INFO] {msg}:", path)


def get_cfg(cfg_filename):
    """获取配置"""
    # 获取当前文件所在目录
    curPath = os.path.dirname(os.path.realpath(__file__))
    # 获取yaml文件路径
    yamlPath = os.path.join(curPath, "config", cfg_filename)

    with open(yamlPath, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader)
    
    return cfg


def get_sift_image(cfg, dataset_root, device, args):
    """筛选高置信度图片"""
    data_name = args.data_name
    model_name = cfg["model_name"]
    model_path = cfg["two_stage_model_path"]
    check_path(model_path, "model_path")
    

    result_path = os.path.join(
        config.output_result, 
        model_name + '-' + data_name,
        "stage3",
        cfg["image_type"]
    )
    print("\n[INFO] result_path:", result_path)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # model
    model = models.load_model(model_name=model_name,
                              in_channels=cfg['model']['in_channels'],
                              num_classes=cfg['model']['num_classes'])
    model.load_state_dict(torch.load(model_path)["model"])
    model.to(device)
    model.eval()

    # data, 普通的数据加载器
    data_loader, _ = loaders.load_data(data_name=data_name, data_type='train')

    image_sift = ImageSift(class_nums=cfg['model']['num_classes'],
                           image_nums=20,
                           is_high_confidence=True)

    # forward
    for i, samples in enumerate(data_loader):
        print('\r[{}/{}]'.format(i, len(data_loader)), end='', flush=True)
        inputs, labels, names = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        image_sift(outputs=outputs, labels=labels, names=names)

    print('\n', end='', flush=True)
    image_sift.save_image(dataset_root, result_path)  # 保存图片

    return os.path.join(result_path, "images")


def find_kernel(cfg, sift_image_path, device, args):

    data_name = args.data_name
    model_name = cfg["model_name"]
    model_path = cfg["two_stage_model_path"]
    check_path(model_path, "model_path")

    input_path = sift_image_path
    result_path = os.path.join(
        config.output_result, 
        model_name + '-' + data_name,
        "stage3",
        "grads"
    )

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # model
    model = models.load_model(model_name=model_name,
                              in_channels=cfg['model']['in_channels'],
                              num_classes=cfg['model']['num_classes'])
    model.load_state_dict(torch.load(model_path)["model"])
    model.eval()
    model.to(device)

    # modules
    modules = models.load_modules(model=model, model_name=model_name, model_layers=None)  # no first conv
    # print("\n modules:", modules)

    grad_sift = GradSift(modules=modules,
                         class_nums=cfg['model']['num_classes'],
                         result_path=result_path)

    data_loader = data_util.load_data(data_path=input_path, data_name=args.data_name)
    for i, samples in enumerate(data_loader):
        print('\r[{}/{}]'.format(i, len(data_loader)), end='', flush=True)
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        grad_sift(outputs, labels)

    grad_sift.sift()

    return result_path

def view_layer_kernel(grad_result_path, cfg, args):
    result_path = grad_result_path
    data_name = args.data_name
    model_name = cfg["model_name"]

    kernel_dict_path = os.path.join(cfg["kernel_dict_path"], f"{model_name}-{data_name}")
    if not os.path.exists(kernel_dict_path):
        os.makedirs(kernel_dict_path)

    # model
    model = models.load_model(model_name=model_name,
                                in_channels=cfg['model']['in_channels'],
                                num_classes=cfg['model']['num_classes'])

    # modules
    modules = models.load_modules(model=model, model_name=model_name, model_layers=None)  # no first conv

    for layer in range(len(modules)):
        label_grads = []
        for label in range(cfg['model']['num_classes']):
            mask_root_path = os.path.join(result_path, str(layer), str(label))
            method_name = 'inputs_label{}_layer{}'.format(label, layer)
            mask_path = os.path.join(mask_root_path, 'grads_{}.npy'.format(method_name))
            data = np.load(mask_path)
            label_grads.append(data)
        
        res_path = os.path.join(kernel_dict_path, "label_grads_layer")
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        np.save(os.path.join(res_path, f"label_grads_layer{layer}.npy"), label_grads)

        pic_path = os.path.join(res_path, f"label_grads_layer{layer}.png")
        view_grads(label_grads, pic_path)


def view_grads(label_grads, pic_path):
    f, ax = plt.subplots(figsize=(64, 32), ncols=1)
    ax.set_xlabel('convolutional kernel')
    ax.set_ylabel('category')
    sns.heatmap(np.array(label_grads).T, ax=ax, linewidths=0.1, annot=False, cbar=False)
    # plt.imshow(np.array(label_grads).T)
    plt.savefig(pic_path, bbox_inches='tight')
    plt.clf()
    plt.close()


def union_cls_kernel(cfg, grad_result_path, args):
    
    result_path = grad_result_path
    data_name = args.data_name
    model_name = cfg["model_name"]

    kernel_dict_path = os.path.join(cfg["kernel_dict_path"], f"{model_name}-{data_name}")
    if not os.path.exists(kernel_dict_path):
        os.makedirs(kernel_dict_path)

    # model
    model = models.load_model(model_name=model_name,
                                in_channels=cfg['model']['in_channels'],
                                num_classes=cfg['model']['num_classes'])

    # modules
    modules = models.load_modules(model=model, model_name=model_name, model_layers=None)  # no first conv

    for idx in range(cfg['model']['num_classes']):
        kernel_dict = {}

        for layer in range(len(modules)):
            for label in range(cfg['model']['num_classes']):
                mask_root_path = os.path.join(result_path, str(layer), str(label))
                method_name = 'inputs_label{}_layer{}'.format(label, layer)
                mask_path = os.path.join(mask_root_path, 'grads_{}.npy'.format(method_name))
                if label == idx:
                    data = np.load(mask_path)
                    # print(f"layer {layer}, label {label}", np.where(data==1))
                    kernel_num = data.size
                    kernel_valid = np.where(np.isin(data, 1))[0].tolist()
                    kernel_val = []
                    kernel_val.append(kernel_num)
                    kernel_val.append(kernel_valid)
                    kernel_dict[layer] = kernel_val

        res_path = os.path.join(kernel_dict_path, f"kernel_dict_label_{idx}.npy")
        np.save(res_path, kernel_dict)

    return kernel_dict_path


def train_and_val(cfg, kernel_dict_root_path, device):
    check_path( kernel_dict_root_path)

    # cfg
    data_name = cfg["data_name"]
    model_name = cfg["model_name"]
    model_path = cfg["model_path"]

    epochs = cfg["epochs"]
    lr = float(cfg["lr"])
    threshold = cfg["threshold"]
    momentum = cfg["momentum"]
    weight_decay = float(cfg["weight_decay"])

    model_layers = range(cfg["model_layer_nums"])
    num_classes=cfg['model']['num_classes']
    in_channels=cfg['model']['in_channels']

    # kernel
    for cls in range(num_classes):
        mask_path_patten = f"{kernel_dict_root_path}/kernel_dict_label_{cls}.npy"
        modify_dict = np.load(mask_path_patten, allow_pickle=True).item()
        modify_dicts.append(modify_dict)
    
    # print(modify_dicts)

    # data
    data_loaders, _ = loaders.load_class_balanced_data(data_name=data_name)

    # model
    model = models.load_model(model_name, in_channels, num_classes)

    # modules
    modules = models.load_modules(model, model_name, model_layers)
    
    # checkpoint
    cp_path = model_path
    check_path(cp_path)
    state = torch.load(cp_path)
   
    model.load_state_dict(state)
    model.to(device)

    loss_fn = reduce_equalized_focal_loss
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = get_lr_scheduler(optimizer, True)

    for epoch in range(epochs):
        epoch_begin_time = time.time()
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        print(f"\nEpoch {epoch+1}")
        print("[INFO] lr is:", cur_lr)
        print("-"*40)
        train(data_loaders["train"], model, loss_fn, optimizer, modules, threshold, modify_dicts, device)
        test(data_loaders["val"], model, loss_fn, optimizer, epoch, threshold, device)
        scheduler.step()

        draw_acc(g_train_loss, g_test_loss, g_train_acc, g_test_acc)
        print_time(time.time()-epoch_begin_time, epoch=True)
        
    print("Done!")


def train(dataloader, model, loss_fn, optimizer, modules, threshold, modify_dicts, device):
    global g_train_loss, g_train_acc
    train_loss, correct = 0, 0
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    # size = len(dataloader.dataset)
    size = 50000 # cifar
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y, _) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        for cls, modify_dict in enumerate(modify_dicts):
            with torch.set_grad_enabled(True):
                # 找到对应类别的图片
                x_pos = (y==cls).nonzero().squeeze()
                # 处理只有一个样本的情况
                if x_pos.shape == torch.Size([]):
                    x_pos = x_pos.unsqueeze(dim=0)
                # 处理没有样本的情况
                if min(x_pos.shape) == 0:
                    continue
                x_cls_i = torch.index_select(X, dim=0, index=x_pos)
                y_cls_i = torch.index_select(y, dim=0, index=x_pos)
                y_train_list.extend(y_cls_i.cpu().numpy())
                # Compute prediction error
                pred = model(x_cls_i)  # 网络前向计算
                loss = loss_fn(pred, y_cls_i, threshold=threshold)
                # loss = focal_loss(pred, y)

                train_loss += loss.item()
            
                y_pred_list.extend(pred.argmax(1).cpu().numpy())

                correct += (pred.argmax(1) == y_cls_i).type(torch.float).sum().item()
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()  # 得到模型中参数对当前输入的梯度
            
                for layer in modify_dict.keys():
                    # if layer <= 19:
                    #     modules[int(layer)].weight.grad[:] = 0
                    # # # print("layer:", layer)
                    for kernel_index in range(modify_dict[layer][0]):
                        if kernel_index not in modify_dict[layer][1]:
                            modules[int(layer)].weight.grad[kernel_index, ::] = 0
            

                optimizer.step()  # 更新参数

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= (num_batches * len(modify_dicts))
    correct /= size
    g_train_loss.append(train_loss)
    g_train_acc.append(correct)
    print("-" * 40)
    print(classification_report(y_train_list, y_pred_list, digits=4))


def test(dataloader, model, loss_fn, optimizer, epoch, threshold, device):
    global best_acc, g_test_loss, g_test_acc
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    for X, y, _ in dataloader:
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)
        with torch.set_grad_enabled(True):
            pred = model(X)
            loss = loss_fn(pred, y, threshold=threshold)
            # loss = focal_loss(pred, y)

            test_loss += loss.item()

        y_pred_list.extend(pred.argmax(1).cpu().numpy())

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    g_test_loss.append(test_loss)
    correct /= size
    g_test_acc.append(correct)
    if correct >= best_acc:
        best_acc = correct
        print(f"[FEAT] epoch {epoch+1}, update best acc:", best_acc)

        best_model_name=f"best-model-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-acc{best_acc:.4f}.pth"
        model_state = {
            'epoch': epoch,  # 注意这里的epoch是从0开始的
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': best_acc,
        }
        torch.save(model_state, best_model_name)

        print(f"Saved Best PyTorch Model State to {best_model_name} \n")
    print(f"Test Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    print(classification_report(y_train_list, y_pred_list, digits=4))


def draw_acc(train_loss, test_loss, train_acc, test_acc):
        num_epochs = len(train_loss)

        plt.plot(range(1, num_epochs + 1), train_loss, 'r', label='train loss')
        plt.plot(range(1, num_epochs + 1), test_loss, 'b', label='val loss')

        plt.plot(range(1, num_epochs + 1), train_acc, 'g', label='train acc')
        plt.plot(range(1, num_epochs + 1), test_acc, 'k', label='val acc')

        plt.title("Acc and Loss of each epoch")
        plt.xlabel("Training Epochs")
        plt.ylabel("Acc & Loss")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.legend()
        plt.savefig('fine_tuning.jpg')
        plt.clf()
        plt.close()


if __name__ == '__main__':
    main()