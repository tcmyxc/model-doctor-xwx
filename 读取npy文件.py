import matplotlib

matplotlib.use('AGG')
import os
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import cv2

import seaborn as sns

np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "%.3g" % x))


pre = np.load("/mnt/nfs/xwx/model-doctor-xwx/output/result/channels/resnet32-cifar-10-lt-ir100/channel_grads_-1.npy")
tail_c = np.load("/mnt/nfs/xwx/model-doctor-xwx/output/result/channels/resnet32-cifar-10-lt-ir100/channels_-1.npy")

threshold= np.mean(pre, axis=1)
# print(threshold)
tail_sum = np.sum(tail_c[-3:], axis=0)
tail_sum = np.where(tail_sum > 0, 1, 0)

for i, (th, grad) in enumerate(zip(threshold, pre)):
    # print(type(grad))
    grad = np.where(grad<th, grad, 0)
    grad = np.where(grad>0, 1, 0)
    # print(grad)
    pre[i] = grad

head_sum = np.sum(pre[:3], axis=0)

kernel = tail_sum + head_sum
kernel = np.where(kernel > 1, 1, 0)
modify_kernel = []
for idx, val in enumerate(kernel):
    if val != 0:
        modify_kernel.append(idx)

print(modify_kernel)
# cur = np.load("/nfs/xwx/model-doctor-xwx/output/result/channels/resnet32-cifar-10-lt-ir100-im2b-epoch30/channel_grads_-1.npy")
# # cur = np.sum(cur, axis=0) / 100
# # print(cur.astype(int))
# # print(cur)
# # plt.plot(cur)
# # plt.savefig("res.png")


# def view_grads(grads, fig_w=64, fig_h=10):
#     f, ax = plt.subplots(figsize=(fig_w, fig_h), ncols=1)
#     ax.set_xlabel('convolutional kernel')
#     ax.set_ylabel('category')
#     sns.heatmap(grads, annot=True, ax=ax, cbar=False)
#     plt.savefig("梯度-epoch30.png", bbox_inches='tight')
#     # plt.show()
#     plt.clf()
#     plt.close()


# view_grads(cur)
