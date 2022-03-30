import os
import yaml
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


best_model_path = None


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
    best_model_path = cp_path
    print(f"Saved Best PyTorch Model State to {model_name} \n")


def draw_acc(result_path, train_loss, test_loss, train_acc, test_acc, filename=None):
    """绘制acc和loss曲线"""
    history = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": test_loss,
        "val_acc": test_acc
    }
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    np.save(os.path.join(result_path, "model.npy" if filename is None else f"{filename}.npy"), history)

    num_epochs = len(train_loss)

    plt.plot(range(1, num_epochs + 1), train_loss, "r", label="train loss")
    plt.plot(range(1, num_epochs + 1), test_loss, "b", label="val loss")

    plt.plot(range(1, num_epochs + 1), train_acc, "g", label="train acc")
    plt.plot(range(1, num_epochs + 1), test_acc, "k", label="val acc")

    plt.title("Acc and Loss of each epoch")
    plt.xlabel("Training Epochs")
    plt.ylabel("Acc & Loss")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "model.jpg" if filename is None else f"{filename}.jpg"))
    plt.clf()
    plt.close()


if __name__ == '__main__':
    draw_acc(
        result_path="./test",
        train_loss=[2,3],
        test_loss=[1,2],
        train_acc=[0.3, 0.9], 
        test_acc=[0.6, 0.5], 
        filename="test"
    )