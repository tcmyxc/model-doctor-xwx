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
        # [module_nums, class_nums, image_nums:grad]

        self.result_path = result_path

    def __call__(self, outputs, labels):
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        for layer, module in enumerate(self.h_modules):
            grads = module.grads(-nll_loss, module.outputs)  # output
            grads = torch.relu(grads)

            # if self.modules[layer][0] == 'Conv2d':
            #     grads = torch.sum(grads, dim=(2, 3))
            grads = torch.sum(grads, dim=(2, 3))

            grads = grads.cpu().numpy()  # GPU to cpu numpy
            for b in range(len(labels)):
                self.grads[layer][labels[b]].append(grads[b])

    def sift(self):
        for layer in range(len(self.modules)):
            # 每个类别在某一层的梯度
            for label, grads in enumerate(self.grads[layer]):
                grads = np.asarray(grads)  # image_nums, val
                # print('grads.shape', grads.shape)

                # grads = np.maximum(grads, 0)  # - to 0
                grads = np.sum(grads, axis=0)  # sum image_nums

                mask = np.zeros(grads.shape)
                threshold = grads.mean()
                mask[np.where(grads > threshold)] = 1

                method_name = 'inputs_label{}_layer{}'.format(label, layer)  # io
                mask_root_path = os.path.join(self.result_path, str(layer), str(label))
                if not os.path.exists(mask_root_path):
                    os.makedirs(mask_root_path)
                mask_path = os.path.join(mask_root_path, 'grads_{}.npy'.format(method_name))
                np.save(mask_path, mask)

                # self.visualize(grads, mask, method_name)
    
    def cal_percent(self):
        np.set_printoptions(threshold=np.inf)
        for layer in range(len(self.modules)):
            # 每个类别在某一层的梯度
            sum_layer_grads = np.zeros(np.asarray(self.grads[layer][0]).shape[-1])
            sum_layer_grads += 1e-10
            # print("sum_layer_grads", sum_layer_grads)
            for label, grads in enumerate(self.grads[layer]):
                grads = np.asarray(grads)  # image_nums, val
                # print('grads.shape', grads.shape)

                # grads = np.maximum(grads, 0)  # - to 0
                grads = np.sum(grads, axis=0)  # sum image_nums
                sum_layer_grads += grads
            
            grad_percent = np.zeros((len(self.grads[layer]), sum_layer_grads.shape[0]))
            for label, grads in enumerate(self.grads[layer]):
                grads = np.asarray(grads)  # image_nums, val
                grads = np.sum(grads, axis=0)  # sum image_nums
                grad_percent[label] = grads / sum_layer_grads
            
            # print("grad_percent", grad_percent)

            method_name = 'inputs_layer{}'.format(layer)  # io
            grad_percent_root_path = os.path.join(self.result_path, "grad_percent")
            if not os.path.exists(grad_percent_root_path):
                os.makedirs(grad_percent_root_path)
            mask_path = os.path.join(grad_percent_root_path, 'grads_percent_{}.npy'.format(method_name))
            np.save(mask_path, grad_percent)

    def visualize(self, grads, mask, method_name):
        print(grads.size)
        grads = grads.reshape((grads.size, -1))
        l_path = os.path.join(self.result_path, 'grads_{}.png'.format(method_name))
        image_util.view_grads(grads.transpose((1, 0)), grads.shape[0], grads.shape[1], l_path)

        mask = mask.reshape((grads.size, -1))
        l_path = os.path.join(self.result_path, 'grads_{}_m.png'.format(method_name))
        image_util.view_grads(mask.transpose((1, 0)), mask.shape[0], mask.shape[1], l_path)


def main():
    data_name = 'cifar-10-lt-ir100'
    model_name = 'resnet32'

    model_path = "/nfs/xwx/model-doctor-xwx/output/model/three-stage/resnet32/cifar-10-lt-ir100/lr0.1/th0.5/custom_lr_scheduler/refl_loss/2022-04-26_20-25-01/best-model-20220426-210920-acc0.7288.pth"
    input_path = "/nfs/xwx/model-doctor-xwx/output/result/resnet32-cifar-10-lt-ir100/stage3/high/images"
    result_path = "/nfs/xwx/model-doctor-xwx/output/result/resnet32-cifar-10-lt-ir100/stage3/grads"

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
    print("\n modules:", modules)

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
