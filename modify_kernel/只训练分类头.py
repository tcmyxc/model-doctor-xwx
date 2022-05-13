"""
冻结特征提取层，只训练分类头
"""
import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

import torch
import torchvision
import models
import loaders
import argparse
import os
import datetime
import time
import matplotlib
import yaml
import math


from torch import optim
from configs import config
from utils.lr_util import get_lr_scheduler
from utils.time_util import print_time, get_current_time
from sklearn.metrics import classification_report
from loss.refl import reduce_equalized_focal_loss
from loss.fl import focal_loss
from loss.hcl import hc_loss
from modify_kernel.util.draw_util import draw_lr
from modify_kernel.util.cfg_util import print_yml_cfg
from functools import partial
from utils.args_util import print_args
from utils.general import init_seeds

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='cifar-10-lt-ir100')
parser.add_argument('--threshold', type=float, default='0.5')
parser.add_argument('--lr', type=float, default='1e-2')
parser.add_argument('--data_loader_type', type=int, default='0', help='0 is default, 1 for cbs')
parser.add_argument('--epochs', type=int, default='200')
parser.add_argument('--lr_scheduler', type=str, default='cosine', help="choose from ['cosine', 'custom', 'constant']")
parser.add_argument('--loss_type', type=str, default='ce', help="choose from ['ce', 'fl', 'refl']")
parser.add_argument('--gpu_id', type=str, default='0')

# global config
best_acc = 0
best_model_path = None
result_path = None
g_train_loss, g_train_acc = [], []
g_test_loss, g_test_acc = [], []
g_cls_test_acc = {}

ft_centers = None

def main():
    args = parser.parse_args()
    print_args(args)

    init_seeds()  # 固定种子

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n[INFO] train on ', device)

    # get cfg
    global result_path, g_cls_test_acc, ft_centers

    threshold = args.threshold
    data_name = args.data_name
    cfg_filename = "cbs_refl.yml"
    cfg = get_cfg(cfg_filename)[data_name]
    print_yml_cfg(cfg) # yml cfg

    for idx in range(cfg['model']['num_classes']):
        g_cls_test_acc[idx] = []

    model_name = cfg["model_name"]
    model_path = cfg["two_stage_model_path"]
    lr = float(args.lr)
    momentum = cfg["optimizer"]["momentum"]
    weight_decay = float(cfg["optimizer"]["weight_decay"])
    epochs = args.epochs
    print(f"\n[INFO] total epoch: {epochs}")
    model_layers = range(cfg["model_layers"])

    # result path
    result_path = os.path.join(config.output_model, "three-stage",
                               model_name, data_name, 
                               f"lr{args.lr}", f"th{args.threshold}",
                               f"{args.lr_scheduler}_lr_scheduler", f"{args.loss_type}_loss",
                               get_current_time())
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    print(f"\n[INFO] result will save in:\n{result_path}\n")

    # data loader
    if args.data_loader_type == 0:
        # 常规数据加载器
        data_loaders, _ = loaders.load_data(data_name=data_name)
    elif args.data_loader_type == 1:
        # 类平衡采样
        data_loaders, _ = loaders.load_class_balanced_data(data_name=data_name)

    # model
    model = models.load_model(
        model_name=model_name,
        in_channels=cfg['model']['in_channels'],
        num_classes=cfg['model']['num_classes']
    )

    model.load_state_dict(torch.load(model_path)["model"])
    model.to(device)

    # 冻结特征层参数
    for i, param in enumerate(model.parameters()):
        if i < 93:
            param.requires_grad = False
    # 随机初始化权重
    print(f"\n[DEBUG] pre linear weight[0]:{model.state_dict()['linear.weight'][0]}")
    stdv = 1. / math.sqrt(model.state_dict()["linear.weight"].size(1))
    model.state_dict()["linear.weight"].data.uniform_(-stdv, stdv)
    print(f"\n[DEBUG] after modify linear weight[0]:{model.state_dict()['linear.weight'][0]}")

    parameters = [p for p in model.parameters() if p.requires_grad]
    # print(f"\n[DEBUG] requires_grad parameters: {parameters}")

    # loss
    if args.loss_type == "ce":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_type == "fl":
        loss_fn = focal_loss
    elif args.loss_type == "refl":
        loss_fn = partial(reduce_equalized_focal_loss, threshold=threshold)

    # optimizer
    optimizer = optim.SGD(
        params=parameters,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        # nesterov=True
    )

    # scheduler
    if args.lr_scheduler == "custom":
        scheduler = get_lr_scheduler(optimizer, True)
    elif args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs
        )
    elif args.lr_scheduler == "constant":
        scheduler = optim.lr_scheduler.ConstantLR(
            optimizer=optimizer,
            factor=1,
            total_iters=epochs
        )

    
    begin_time = time.time()
    lr_list=[]
    for epoch in range(epochs):
        epoch_begin_time = time.time()
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        lr_list.append(cur_lr)
        print(f"\nEpoch {epoch+1}")
        print("[INFO] lr is:", cur_lr)
        print("-" * 42)

        train(data_loaders["train"], model, loss_fn, optimizer, model_layers, device)
        test(data_loaders["val"], model, loss_fn, optimizer, scheduler, epoch, device)
        scheduler.step()

        draw_acc(g_train_loss, g_test_loss, g_train_acc, g_test_acc, result_path)
        draw_lr(result_path, lr_list)
        print_time(time.time()-epoch_begin_time, epoch=True)
        
    print("Done!")
    print_time(time.time()-begin_time)


def train(dataloader, model, loss_fn, optimizer, model_layers, device):
    global g_train_loss, g_train_acc, result_path
    train_loss, correct = 0, 0
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    features = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()
    for batch, (X, y, _) in enumerate(dataloader):
        y_train_list.extend(y.numpy())
        X, y = X.to(device), y.to(device)
        
        with torch.set_grad_enabled(True):
            # Compute prediction error
            pred, feature_out = model(X)  # 网络前向计算

            # 存储特征图，用于TSEN聚类
            tmp_feature_out = feature_out.detach().cpu()
            tmp_feature_out = torch.flatten(tmp_feature_out, 1)
            features.extend(tmp_feature_out.numpy())
            
            fn_loss = loss_fn(pred, y)
            loss_hc = hc_loss(pred, y)
            loss = fn_loss + loss_hc

            train_loss += loss.item()
        
            y_pred_list.extend(pred.argmax(1).cpu().numpy())

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()  # 得到模型中参数对当前输入的梯度
            optimizer.step()  # 更新参数
                

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"[{current:>5d}/{size:>5d}] loss: {loss:>7f}, fn_loss: {fn_loss:>7f}", flush=True)

    train_loss /= num_batches
    print(f"\n[DEBUG] train_loss: {train_loss:>7f}")
    
    correct /= size
    g_train_loss.append(train_loss)
    g_train_acc.append(correct)
    print("-" * 42)
    print(classification_report(y_train_list, y_pred_list, digits=4))

    # 特征图可视化
    # draw_tsne(result_path, features, "train", y_train_list)


def test(dataloader, model, loss_fn, optimizer, scheduler, epoch, device):
    global best_acc, g_test_loss, g_test_acc, result_path
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    features = []
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    for batch, (X, y, _) in enumerate(dataloader):
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)
        with torch.set_grad_enabled(True):
            pred, feature_out = model(X)  # 网络前向计算

            tmp_feature_out = feature_out.detach().cpu()
            tmp_feature_out = torch.flatten(tmp_feature_out, 1)
            features.extend(tmp_feature_out.numpy())

            loss = loss_fn(pred, y)
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
    is_best = (correct > best_acc)
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
    draw_classification_report("test", result_path, y_train_list, y_pred_list, is_best)

    # 特征图可视化
    # draw_tsne(result_path, features, "val", y_train_list)
    


def draw_tsne(result_path, features, mode, y_train_list):
    # 特征图可视化
    from sklearn.manifold import TSNE
    # colorBoard=["dimgray","darkorange","tan","silver","forestgreen",\
    #             "darkgreen","royalblue","navy","red","darksalmon","peru","olive",\
    #             "yellow","cyan","mediumaquamarine","skyblue","purple","fuchsia",\
    #             "indigo","khaki"]

    colors = np.array(["C0","C1","C2","C3","C4","C5","C6","C7", "C8","C9"])
    # colors = np.array(colorBoard)

    features_embedded = TSNE(n_components=2, init='pca', n_iter=1000).fit_transform(features)
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=colors[y_train_list])
    # plt.colorbar()
    plt.title(f"{mode} dataset tsne")
    plt.savefig(os.path.join(result_path, f"{mode}_tsne.jpg"))
    plt.clf()
    plt.close()


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
    plt.savefig(os.path.join(result_path, "model.jpg"))
    plt.clf()
    plt.close()


def draw_classification_report(mode_type, result_path, y_train_list, y_pred_list, is_best):
    """绘制在训练集/测试集上面的 classification_report"""
    reports = classification_report(y_train_list, y_pred_list, digits=4, output_dict=True)
    np.save(os.path.join(result_path, f"{mode_type}_classification_report.npy"), reports)

    labels = []
    accs = []
    samplers =[]
    for x_i, y_i in reports.items():
        if x_i == "accuracy": break
        # print(f"\n[DEBUG] type: {type(x_i)}")
        labels.append(float(x_i))
        accs.append(float(y_i["recall"]))
        samplers.append(float(y_i["support"]))

    # 如果得到了更好的模型，再绘制每个类别的ACC变化曲线
    if is_best:
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

    np.save(os.path.join(result_path, "cls_test_acc_report.npy"), g_cls_test_acc)
    
    num_epochs = len(g_cls_test_acc[0])

    for label in labels:
        plt.plot(range(1, num_epochs + 1), g_cls_test_acc[int(label)], label=f"label {label}")

    plt.xlabel("Training Epochs")
    plt.ylabel("Acc")
    # plt.legend(loc="best")
    # plt.legend(loc="lower left")
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