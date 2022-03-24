import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    plt.legend()
    plt.savefig(os.path.join(result_path, "model.jpg" if filename is None else f"{filename}.jpg"))
    plt.clf()
    plt.close()


if __name__ == "__main__":
    pass