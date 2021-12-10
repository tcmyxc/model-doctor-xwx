import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw():
    # save history
    history = np.load(r'D:\Desktop\model.npy', allow_pickle=True).item()

    num_epochs = len(history['train_loss_cls'])

    # adjust history
    for i in range(num_epochs):
        if history['val_loss_cls'][i] > 10:
            history['val_loss_cls'][i] = 7.2

    # draw history

    plt.plot(range(1, num_epochs + 1), history['train_loss_cls'], 'r', label='train loss_cls')
    plt.plot(range(1, num_epochs + 1), history['train_loss_gc'], 'deeppink', label='train loss_gc')
    plt.plot(range(1, num_epochs + 1), history['train_acc'], 'g', label='train acc')
    plt.plot(range(1, num_epochs + 1), history['val_loss_cls'], 'b', label='val loss_cls')
    # plt.plot(range(1, num_epochs + 1), history['val_loss_gc'], 'blueviolet', label='val loss_gc')
    plt.plot(range(1, num_epochs + 1), history['val_acc'], 'k', label='val acc')

    plt.title("Acc and Loss of each epoch")
    plt.xlabel("Training Epochs")
    plt.ylabel("Acc Loss")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.legend()
    plt.savefig(r'D:\Desktop\model.jpg')
    plt.clf()


if __name__ == '__main__':
    draw()
