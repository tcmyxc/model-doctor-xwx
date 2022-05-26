import numpy as np

modify_dict = np.load("/nfs/xwx/model-doctor-xwx/output/result/resnet32-cifar-10-lt-ir100/stage3/grads/grad_mask/grad_mask_layer_0.npy", allow_pickle=True)

for i in modify_dict:
    print(i)