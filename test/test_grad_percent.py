import matplotlib

matplotlib.use('AGG')
import os
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import cv2

import seaborn as sns

percent_path = "/nfs/xwx/model-doctor-xwx/output/result/resnet32-imagenet-10-lt/stage3/grads/grad_percent/grads_percent_inputs_layer10.npy"

modify_dict = np.load(percent_path)
print(modify_dict.shape)


# f, ax = plt.subplots(figsize=(28, 10), ncols=1)
# sns.heatmap(modify_dict, annot=False, ax=ax, linewidths=0.1, fmt=".6f")
# plt.savefig("percent.png", bbox_inches='tight')
# # plt.show()
# plt.clf()
# plt.close()