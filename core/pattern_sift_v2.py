"""
grads
"""
import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

import torch
import numpy as np
import os
import json

import models
from configs import config
from utils import image_util
from utils import data_util



class HookModule:
    def __init__(self, module):
        self.inputs = None
        self.outputs = None
        module.register_forward_hook(self._hook)

    def grads(self, outputs, inputs=None, retain_graph=True, create_graph=False):
        if inputs is None:
            inputs = self.inputs

        return torch.autograd.grad(outputs=outputs,
                                   inputs=inputs,
                                   retain_graph=retain_graph,
                                   create_graph=create_graph)[0]

    def _hook(self, module, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


class GradSift:
    def __init__(self, modules, class_nums, result_path):
        self.h_modules = [HookModule(module) for module in modules]
        self.modules = modules

        self.grads = [[[] for _ in range(class_nums)] for _ in range(len(modules))]

        self.result_path = result_path

    def __call__(self, outputs, labels):
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        for layer, module in enumerate(self.h_modules):
            grads = module.grads(-nll_loss, module.outputs)  # output
            grads = torch.relu(grads)
            grads = torch.sum(grads, dim=(2, 3))

            grads = grads.cpu().numpy()  # GPU to cpu numpy
            for b in range(len(labels)):
                self.grads[layer][labels[b]].append(grads[b])

    def sift(self):
        for layer in range(len(self.modules)):
            # 每个类别在某一层的梯度
            layer_masks = []
            for label, grads_i in enumerate(self.grads[layer]):
                grads_i = np.asarray(grads_i)  # image_nums, val
                grads_i = np.sum(grads_i, axis=0)  # sum image_nums

                mask_i = np.zeros(grads_i.shape)
                threshold = grads_i.mean()
                mask_i[np.where(grads_i > threshold)] = 1
                layer_masks.append(mask_i)
            
            layer_masks_sum = 0
            print()
            for i in layer_masks:
                # print(i)
                layer_masks_sum += i
            
            # print(type(layer_masks_sum))
            mask_root_path = os.path.join(self.result_path, "grad_mask")
            if not os.path.exists(mask_root_path):
                os.makedirs(mask_root_path)
            mask_path = os.path.join(mask_root_path, f'grad_mask_layer_{layer}.npy')
            np.save(mask_path, layer_masks_sum)
            # break

                



def main():
    data_name = 'cifar-10'
    model_name = 'resnet32'

    model_path = "/nfs/xwx/model-doctor-xwx/output/model/pretrained/resnet32/cifar-10/lr0.01/cosine_lr_scheduler/ce_loss/best-model.pth"
    input_path = "/nfs/xwx/model-doctor-xwx/output/result/resnet32-cifar-10/high/images"
    result_path = "/nfs/xwx/model-doctor-xwx/output/result/resnet32-cifar-10/grads"

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # config
    cfg = json.load(open('configs/config_trainer.json'))[data_name]
    device = torch.device('cuda:0')

    # model
    model = models.load_model(model_name=model_name,
                              in_channels=cfg['model']['in_channels'],
                              num_classes=cfg['model']['num_classes'])
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    model.to(device)

    # modules
    modules = models.load_modules(model=model, model_name=model_name, model_layers=None)  # no first conv
    # print("\n modules:", modules)

    grad_sift = GradSift(modules=modules,
                         class_nums=cfg['model']['num_classes'],
                         result_path=result_path)

    data_loader = data_util.load_data(input_path, data_name)
    for i, samples in enumerate(data_loader):
        print('\r[{}/{}]'.format(i, len(data_loader)), end='', flush=True)
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs, _ = model(inputs)
        grad_sift(outputs, labels)

    grad_sift.sift()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    np.set_printoptions(threshold=np.inf)

    main()
