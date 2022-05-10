import sys
sys.path.append('/nfs/xwx/model-doctor-xwx') #205

import os
import matplotlib
import models
import json

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    data_name = 'cifar-10'
    model_name = 'resnet32'
    grad_result_path = "/nfs/xwx/model-doctor-xwx/output/result/resnet32-cifar-10/grads"

    # config
    cfg = json.load(open('configs/config_trainer.json'))[data_name]
    
    view_layer_kernel(grad_result_path, cfg, model_name, data_name)


def view_layer_kernel(grad_result_path, cfg, model_name, data_name):
    result_path = grad_result_path
    kernel_dict_path = "/nfs/xwx/model-doctor-xwx/modify_kernel/kernel_dict"

    kernel_dict_path = os.path.join(kernel_dict_path, f"{model_name}-{data_name}")
    if not os.path.exists(kernel_dict_path):
        os.makedirs(kernel_dict_path)

    # model
    model = models.load_model(model_name=model_name,
                                in_channels=cfg['model']['in_channels'],
                                num_classes=cfg['model']['num_classes'])

    # modules
    modules = models.load_modules(model=model, model_name=model_name, model_layers=None)  # no first conv

    for layer in range(len(modules)):
        label_grads = []
        for label in range(cfg['model']['num_classes']):
            mask_root_path = os.path.join(result_path, str(layer), str(label))
            method_name = 'inputs_label{}_layer{}'.format(label, layer)
            mask_path = os.path.join(mask_root_path, 'grads_{}.npy'.format(method_name))
            data = np.load(mask_path)
            label_grads.append(data)
        
        res_path = os.path.join(kernel_dict_path, "label_grads_layer")
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        np.save(os.path.join(res_path, f"label_grads_layer{layer}.npy"), label_grads)

        pic_path = os.path.join(res_path, f"label_grads_layer{layer}.png")
        view_grads(label_grads, pic_path)


def view_grads(label_grads, pic_path):
    f, ax = plt.subplots(figsize=(28, 10), ncols=1)
    ax.set_xlabel('convolutional kernel')
    ax.set_ylabel('category')
    # sns.heatmap(np.array(label_grads), ax=ax, linewidths=0.1, annot=False, cbar=False)
    sns.heatmap(np.array(label_grads), ax=ax, linewidths=0.1, annot=False)
    # plt.imshow(np.array(label_grads).T)
    plt.savefig(pic_path, bbox_inches='tight')
    plt.clf()
    plt.close()



if __name__ == '__main__':
    main()