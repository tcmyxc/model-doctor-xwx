import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

import torch
import models
import loaders
import argparse
import os
import datetime
import time
import matplotlib
import yaml

from torch import optim
from configs import config
from utils.lr_util import get_lr_scheduler
from utils.time_util import print_time, get_current_time
from sklearn.metrics import classification_report
from loss.refl import reduce_equalized_focal_loss
from hooks.grad_hook import GradHookModule

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='imagenet-10-lt')
parser.add_argument('--threshold', type=float, default='0.5')
parser.add_argument('--lr', type=float, default='1e-2')


# global config
modify_dicts = []
# kernel_percents = {}
threshold = None
best_acc = 0
best_model_path = None
result_path = None
g_train_loss, g_train_acc = [], []
g_test_loss, g_test_acc = [], []
g_cls_test_acc = {}

def main():
    args = parser.parse_args()
    print(f"\n[INFO] args: {args}")

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n[INFO] train on ', device)

    # get cfg
    global threshold, result_path, g_cls_test_acc
    threshold = args.threshold
    data_name = args.data_name
    cfg_filename = "cbs_refl.yml"
    cfg = get_cfg(cfg_filename)[data_name]

    print("-" * 42)
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("-" * 42)

    for idx in range(cfg['model']['num_classes']):
        g_cls_test_acc[idx] = []

    model_name = cfg["model_name"]
    model_path = cfg["two_stage_model_path"]
    # lr = float(cfg["optimizer"]["lr"])
    lr = float(args.lr)
    momentum = cfg["optimizer"]["momentum"]
    weight_decay = float(cfg["optimizer"]["weight_decay"])
    # epochs = cfg["three_stage_epochs"]
    epochs = 200
    print(f"\n[INFO] total epoch: {epochs}")
    model_layers = range(cfg["model_layers"])

    # result path
    result_path = os.path.join(config.output_model, "three-stage",
                               model_name, data_name, 
                               f"lr{args.lr}", f"th{args.threshold}",
                               get_current_time())
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # kernel
    num_classes = cfg['model']['num_classes']
    kernel_dict_path = os.path.join(
        cfg["kernel_dict_path"],
        f"{model_name}-{data_name}"
    )
    # 01mask
    for cls in range(num_classes):
        mask_path_patten = f"{kernel_dict_path}/kernel_dict_label_{cls}.npy"
        modify_dict = np.load(mask_path_patten, allow_pickle=True).item()
        modify_dicts.append(modify_dict)

    # data
    data_loaders, _ = loaders.load_data(data_name=data_name)

    # model
    model = models.load_model(
        model_name=model_name,
        in_channels=cfg['model']['in_channels'],
        num_classes=cfg['model']['num_classes']
    )

    # modules
    modules = models.load_modules(
        model=model,
        model_name=model_name,
        model_layers=model_layers
    )

    model.load_state_dict(torch.load(model_path)["model"])
    model.to(device)

    # optimizer
    loss_fn = reduce_equalized_focal_loss
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    # scheduler = get_lr_scheduler(optimizer, True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs
    )

    train(data_loaders["train"], model, loss_fn, optimizer, modules, device)



def train(dataloader, model, loss_fn, optimizer, modules, device):
    # model.train()
    model.eval()
    features = []
    y_pred_list = []
    y_train_list = []
    for batch, (X, y, _) in enumerate(dataloader):
        y_train_list.extend(y.numpy())
        X, y = X.to(device), y.to(device)
    #     for cls, modify_dict in enumerate(modify_dicts):
    #         with torch.set_grad_enabled(True):
    #             # 找到对应类别的图片
    #             x_pos = (y==cls).nonzero().squeeze()
    #             # 处理只有一个样本的情况
    #             if x_pos.shape == torch.Size([]):
    #                 x_pos = x_pos.unsqueeze(dim=0)
    #             # 处理没有样本的情况
    #             if min(x_pos.shape) == 0:
    #                 continue
    #             x_cls_i = torch.index_select(X, dim=0, index=x_pos)
    #             y_cls_i = torch.index_select(y, dim=0, index=x_pos)
    #             y_train_list.extend(y_cls_i.cpu().numpy())
                
    #             # Compute prediction error
        _, feature_out = model(X)  # 网络前向计算
        feature_out = feature_out.detach().cpu()
        feature_out = torch.flatten(feature_out, 1)
        # print("feature_out", feature_out.shape)
        features.extend(feature_out.numpy())
    
    from sklearn.manifold import TSNE
    features = np.array(features)
    # np.save("features.npy", features)
    # print(features.shape)
    # for iter_num in range(250, 10000):
    features_embedded = TSNE(n_components=2, init='pca', n_iter=1000).fit_transform(features)
    
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=y_train_list)
    # features_embedded = TSNE(n_components=3, init='pca', n_iter=5000).fit_transform(features)
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(features_embedded[:, 0], features_embedded[:, 1], features_embedded[:, 2], c=y_train_list)
    plt.colorbar()
    plt.savefig("tsne.jpg")
    plt.clf()
    plt.close()



def test(dataloader, model, loss_fn, optimizer, scheduler, epoch, device):
    global best_acc, g_test_loss, g_test_acc, result_path
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    for batch, (X, y, _) in enumerate(dataloader):
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)
        with torch.set_grad_enabled(True):
            pred, _ = model(X)  # 网络前向计算
            loss = loss_fn(pred, y, threshold=threshold)

            test_loss += loss.item()

        y_pred_list.extend(pred.argmax(1).cpu().numpy())

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)

    test_loss /= num_batches
    g_test_loss.append(test_loss)
    correct /= size
    g_test_acc.append(correct)
    if correct > best_acc:
        best_acc = correct
        print(f"\n[FEAT] epoch {epoch+1}, update best acc: {best_acc:.4f}")
        model_name=f"best-model-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-acc{best_acc:.4f}.pth"
        model_state = {
            'epoch': epoch,  # 注意这里的epoch是从0开始的
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            'acc': best_acc
        }
        update_best_model(result_path, model_state, model_name)

    print(f"\n[INFO] Test Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    print(classification_report(y_train_list, y_pred_list, digits=4))
    draw_classification_report("test", result_path, y_train_list, y_pred_list)


def draw_acc(train_loss, test_loss, train_acc, test_acc, result_path):
    history = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': test_loss,
        'val_acc': test_acc
    }

    np.save(os.path.join(result_path, 'model.npy'), history)

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
    plt.savefig(os.path.join(result_path, "three_stage_model.jpg"))
    plt.clf()
    plt.close()


def draw_classification_report(mode_type, result_path, y_train_list, y_pred_list):
    """绘制在训练集/测试集上面的 classification_report"""
    reports = classification_report(y_train_list, y_pred_list, digits=4, output_dict=True)
    np.save(os.path.join(result_path, f"{mode_type}_classification_report.npy"), reports)

    labels = []
    accs = []
    samplers =[]
    for x_i, y_i in reports.items():
        if x_i == "accuracy": break
        labels.append(x_i)
        accs.append(y_i["recall"])
        samplers.append(y_i["support"])

    draw_cls_test_acc(labels, accs, result_path)
    plt.plot(labels, accs)
    plt.title("Acc of each class")
    plt.xlabel("Classes")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(result_path, f"{mode_type}_classification_report.jpg"))
    plt.clf()
    plt.close()


def draw_cls_test_acc(labels, one_epoch_test_acc, result_path):
    global g_cls_test_acc
    for idx in labels:
        g_cls_test_acc[int(idx)].append(one_epoch_test_acc[int(idx)])

    num_epochs = len(g_cls_test_acc[0])

    for label in labels:
        plt.plot(range(1, num_epochs + 1), g_cls_test_acc[int(label)], label=f"label {label}")

    plt.xlabel("Training Epochs")
    plt.ylabel("Acc")
    # plt.legend(loc="best")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "cls_test_acc_report.jpg"))
    plt.clf()
    plt.close()


def get_cfg(cfg_filename):
    """获取配置"""
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    # 获取当前文件所在目录
    curPath = os.path.dirname(os.path.realpath(__file__))
    # 获取yaml文件路径
    yamlPath = os.path.join(curPath, "config", cfg_filename)

    with open(yamlPath, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader)
    
    return cfg


def update_best_model(result_path, model_state, model_name):
    """更新权重文件"""
    global best_model_path
    cp_path = os.path.join(result_path, model_name)

    if best_model_path is not None:
        # remove previous model weights
        os.remove(best_model_path)

    torch.save(model_state, cp_path)
    torch.save(model_state, os.path.join(result_path, "best-model.pth"))
    best_model_path = cp_path
    print(f"Saved Best PyTorch Model State to {model_name} \n")


if __name__ == '__main__':
    main()