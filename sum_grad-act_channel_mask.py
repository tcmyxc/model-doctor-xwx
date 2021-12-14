import numpy as np
import os
channels_path = "/home/xwx/model-doctor-xwx/output/result/channels/resnet50-cifar-10"
grads_path = os.path.join(channels_path, 'channels_-1-pos-grad.npy')
channels_grads = np.load(grads_path)   # (Class, channel)

print(channels_grads)
act_path = os.path.join(channels_path, 'channels_-1-pos-act.npy')
channels_act = np.load(act_path)

# 并集
channel_u = channels_grads + channels_act
np.save(channels_path + "channels_-1-u.npy", channel_u)

# 交集
channel_n = channels_grads * channels_act
np.save(channels_path + "channels_-1-n.npy", channel_n)