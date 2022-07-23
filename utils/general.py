"""
General utils
"""

import contextlib
import glob
import logging
import math
import os
import platform
import random
import re
import shutil
import signal
import time
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml


def init_seeds(seed=0):
    """固定随机种子"""
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)
    
    
def update_best_model(cfg, model_state, model_name):
    """更新权重文件"""

    result_path = cfg["result_path"]
    cp_path = os.path.join(result_path, model_name)

    if cfg["best_model_path"] is not None:
        # remove previous model weights
        os.remove(cfg["best_model_path"])

    torch.save(model_state, cp_path)
    torch.save(model_state, os.path.join(result_path, "best-model.pth"))
    cfg["best_model_path"] = cp_path
    print(f"Saved Best PyTorch Model State to {model_name} \n")


def get_head_and_kernel(channel_path, head_ratio=0.3):
    """获取需要修改的头部类和卷积核"""
    modify_dict = np.load(channel_path)
    cls_num = int(len(modify_dict) * head_ratio)
    
    head_num = int(len(modify_dict) * 0.3)
    head = [i for i in range(head_num)]
        
    head_sum = np.sum(modify_dict[:cls_num], axis=0)
    head_sum = np.where(head_sum > 0, 1, 0)

    tail_sum = np.sum(modify_dict[-cls_num:], axis=0)
    tail_sum = np.where(tail_sum > 0, 1, 0)

    kernel = tail_sum + head_sum
    kernel = np.where(kernel > 1, 1, 0)
    modify_kernel = []
    for idx, val in enumerate(kernel):
        if val != 0:
            modify_kernel.append(idx)
    
    return modify_kernel, head


def get_head_ratio(data_name):
    head_ratio = None
    if data_name == "cifar-10-lt-ir10":
        head_ratio = 0.3
    elif data_name == "cifar-10-lt-ir100":
        head_ratio = 0.3
    elif data_name == "cifar-100-lt-ir10":
        head_ratio = 0.05
    elif data_name == "cifar-100-lt-ir50":
        head_ratio = 0.05
    elif data_name == "cifar-100-lt-ir100":
        head_ratio = 0.05
    
    return head_ratio

if __name__ == '__main__':
    init_seeds()