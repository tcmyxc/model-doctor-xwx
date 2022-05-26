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
from utils.time_util import print_time
from loss.refl import reduce_equalized_focal_loss
from sklearn.metrics import classification_report

def view_grads(label_grads, pic_path):
    f, ax = plt.subplots(figsize=(64, 10), ncols=1)
    ax.set_xlabel('convolutional kernel')
    ax.set_ylabel('category')
    # sns.heatmap(np.array(label_grads), ax=ax, linewidths=0.1, annot=False, cbar=False)
    sns.heatmap(np.array(label_grads), ax=ax, linewidths=0.1, annot=True, fmt=".3f")
    # plt.imshow(np.array(label_grads).T)
    plt.savefig(pic_path, bbox_inches='tight')
    plt.clf()
    plt.close()

path = "/nfs/xwx/model-doctor-xwx/output/result/resnet32-cifar-10/grads/grad_mask/grad_mask_layer_29.npy"
label_grads = np.load(path)
print(label_grads)
# view_grads(label_grads, "grad_percnet.png")

