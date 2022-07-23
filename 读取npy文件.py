import matplotlib

matplotlib.use('AGG')
import os
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import cv2

import seaborn as sns

np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "%.3g" % x))


pre = np.load("/nfs/xwx/model-doctor-xwx/output/result/channels/resnet32-cifar-10-lt-ir100-ori/channel_grads_-1.npy")
# print(pre.astype(int))
cur = np.load("/nfs/xwx/model-doctor-xwx/output/result/channels/resnet32-cifar-10-lt-ir100-im2b-epoch30/channel_grads_-1.npy")
# cur = np.sum(cur, axis=0) / 100
# print(cur.astype(int))
# print(cur)
# plt.plot(cur)
# plt.savefig("res.png")


def view_grads(grads, fig_w=64, fig_h=10):
    f, ax = plt.subplots(figsize=(fig_w, fig_h), ncols=1)
    ax.set_xlabel('convolutional kernel')
    ax.set_ylabel('category')
    sns.heatmap(grads, annot=True, ax=ax, cbar=False)
    plt.savefig("梯度-epoch30.png", bbox_inches='tight')
    # plt.show()
    plt.clf()
    plt.close()


view_grads(cur)
