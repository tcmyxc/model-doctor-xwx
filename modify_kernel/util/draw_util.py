import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report


def draw_acc_and_loss(train_loss, test_loss, 
                      train_acc, test_acc, 
                      result_path, filename=None):
    """绘制acc和loss曲线"""
    
    history = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": test_loss,
        "val_acc": test_acc
    }
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    np.save(os.path.join(result_path, "model_acc_loss.npy" if filename is None else f"{filename}.npy"), history)

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
    plt.savefig(os.path.join(result_path, "model_acc_loss.jpg" if filename is None else f"{filename}.jpg"))
    plt.clf()
    plt.close()


def draw_lr(result_path, lr_list):
    """绘制lr曲线"""

    num_epochs = len(lr_list)

    plt.plot(range(1, num_epochs + 1), lr_list, label='lr')

    plt.title("Learning rate of each epoch")
    plt.xlabel("Training Epochs")
    plt.ylabel("Learning rate")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "model_lr.jpg"))
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

    plt.plot(labels, accs)
    plt.title("Acc of each class")
    plt.xlabel("Classes")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(result_path, f"{mode_type}_classification_report.jpg"))
    plt.clf()
    plt.close()


def draw_fc_weight(result_path, fc_weight):
    """绘制 fc weight"""

    fc_weight = np.sum(fc_weight * fc_weight, axis=1)
    fc_weight = fc_weight**0.5

    plt.plot(range(len(fc_weight)), fc_weight, 'r', label='fc _weight')
    
    plt.title("fc weight")
    plt.xlabel("class")
    plt.ylabel("l2 weight")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "fc_weight.png"))
    plt.clf()
    plt.close()
        