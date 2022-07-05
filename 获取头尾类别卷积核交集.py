import numpy as np
import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')
from utils.general import get_head_and_kernel
np.set_printoptions(linewidth=100000)

channel_path = "/nfs/xwx/model-doctor-xwx/output/result/channels/resnet32-cifar-10-lt-ir10/channels_-1.npy"

modify_dict = np.load(channel_path)
cls_num = int(len(modify_dict) * 0.3)
    
head_sum = np.sum(modify_dict[:cls_num], axis=0)
head_sum = np.where(head_sum > 0, 1, 0)

tail_sum = np.sum(modify_dict[-cls_num:], axis=0)
tail_sum = np.where(tail_sum > 0, 1, 0)

kernel = tail_sum + head_sum
kernel = np.where(kernel > 1, 1, 0)
modify_kernel = []
for idx, val in enumerate(kernel):
    if val != 0:
        modify_kernel.append(idx)

print(modify_kernel, len(modify_kernel))
print(get_head_and_kernel(channel_path))

