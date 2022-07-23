from torchvision import datasets
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# my_dataset = datasets.ImageFolder(root="/nfs/xwx/dataset/cifar100_lt_ir10/images/train")
# print(my_dataset.classes)
# print(my_dataset.class_to_idx)

def draw_fc_weight_history(fc_weight_list):
    """绘制 fc weight 变化情况"""

    num_epochs = len(fc_weight_list)
    W = np.concatenate(fc_weight_list)
    W = W.reshape((num_epochs, -1))

    plt.figure(figsize=(4, 6))
    plt.imshow(W, cmap='jet')
    plt.colorbar()
    plt.xlabel('class ID')
    plt.ylabel('training epochs')
    plt.title('L2 norm')
    
    plt.savefig("fc_weight_history.png")
    plt.clf()
    plt.close()
    
    
path_to_fc_weight_list = "/nfs/xwx/model-doctor-xwx/output/model/three-stage/resnet32/cifar-100-lt-ir10/lr0.01/th0.5/custom_lr_scheduler/ce_loss/2022-07-07_01-00-25/fc_weight_history.npy"

draw_fc_weight_history(np.load(path_to_fc_weight_list))