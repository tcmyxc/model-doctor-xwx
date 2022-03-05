# 功能概述
# 1. 使用stage2训练好的模型挑选高置信度图片
# 2. 使用高置信度图片找类别有关的卷积核
# 3. 训练模型

import sys
import os
import torch
import json
import yaml
import numpy as np
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

sys.path.append('/nfs/xwx/model-doctor-xwx') #205
import loaders
import models
from configs import config
from utils import data_util
from core.image_sift import ImageSift
from core.pattern_sift import GradSift

# todo：再写一个函数，只是普通加载数据，不使用类别平衡采样

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device('cuda:0')
    cfg = get_cfg()
    sift_image_path = get_sift_image(cfg, device)
    # grad_result_path = find_kernel(cfg, sift_image_path, device)
    # union_cls_kernel(cfg, grad_result_path)


def check_path(path, msg=None):
    """检查路径是否合法"""
    if not os.path.exists(path):
        if msg == None:
            print("\n[ERROR] path does not exist")
        else:
            print(f"\n[ERROR] {msg} does not exist")
        return
    else:
        if msg == None:
            print("\n[INFO] path:", path)
        else:
            print(f"\n[INFO] {msg}:", path)


def get_cfg():
    """获取配置"""
    # 获取当前文件所在目录
    curPath = os.path.dirname(os.path.realpath(__file__))
    # 获取yaml文件路径
    yamlPath = os.path.join(curPath, "config", "cifar_10_lt_ir100.yml")

    with open(yamlPath, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader)
    
    return cfg


def get_sift_image(cfg, device):
    """筛选高置信度图片"""
    data_name = cfg["data_name"]
    model_name = cfg["model_name"]
    model_path = cfg["model_path"]
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

    # sift_image(data_name, model_name, model_path, result_path)

    # config
    local_cfg = json.load(open('configs/config_trainer.json'))[data_name]

    # model
    model = models.load_model(model_name=model_name,
                              in_channels=local_cfg['model']['in_channels'],
                              num_classes=local_cfg['model']['num_classes'])
    model.load_state_dict(torch.load(model_path)["model"])
    model.to(device)
    model.eval()

    # data，训练集
    data_loader, _ = loaders.load_data(data_name=data_name, data_type='train')

    image_sift = ImageSift(class_nums=local_cfg['model']['num_classes'],
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
    image_sift.save_image(result_path)  # 保存图片

    return os.path.join(result_path, "images")


def find_kernel(cfg, sift_image_path, device):
    data_name = cfg["data_name"]
    model_name = cfg["model_name"]
    model_path = cfg["model_path"]
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

    # config
    local_cfg = json.load(open('configs/config_trainer.json'))[data_name]

    # model
    model = models.load_model(model_name=model_name,
                              in_channels=local_cfg['model']['in_channels'],
                              num_classes=local_cfg['model']['num_classes'])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    # modules
    modules = models.load_modules(model=model, model_name=model_name, model_layers=None)  # no first conv
    # print("\n modules:", modules)

    grad_sift = GradSift(modules=modules,
                         class_nums=local_cfg['model']['num_classes'],
                         result_path=result_path)

    data_loader = data_util.load_data(input_path)
    for i, samples in enumerate(data_loader):
        print('\r[{}/{}]'.format(i, len(data_loader)), end='', flush=True)
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        grad_sift(outputs, labels)

    grad_sift.sift()

    return result_path


def union_cls_kernel(cfg, grad_result_path):
    result_path = grad_result_path
    data_name = cfg["data_name"]
    model_name = cfg["model_name"]

    kernel_dict_path = os.path.join(cfg["kernel_dict_path"], f"{model_name}-{data_name}")
    if not os.path.exists(kernel_dict_path):
        os.makedirs(kernel_dict_path)

    # config
    local_cfg = json.load(open('configs/config_trainer.json'))[data_name]
    # model
    model = models.load_model(model_name=model_name,
                                in_channels=local_cfg['model']['in_channels'],
                                num_classes=local_cfg['model']['num_classes'])

    # modules
    modules = models.load_modules(model=model, model_name=model_name, model_layers=None)  # no first conv

    for idx in range(local_cfg['model']['num_classes']):
        kernel_dict = {}

        for layer in range(len(modules)):
            for label in range(local_cfg['model']['num_classes']):
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

if __name__ == '__main__':
    main()