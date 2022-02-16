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
        self.h_modules = [HookModule(module[1]) for module in modules]
        self.modules = modules

        self.grads = [[[] for _ in range(class_nums)] for _ in range(len(modules))]
        # [module_nums, class_nums, image_nums:grad]

        self.result_path = result_path

    def __call__(self, outputs, labels):
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        for layer, module in enumerate(self.h_modules):
            grads = module.grads(-nll_loss, module.inputs)  # io
            grads = torch.relu(grads)

            if self.modules[layer][0] == 'Conv2d':
                grads = torch.sum(grads, dim=(2, 3))

            grads = grads.cpu().numpy()  # GPU to cpu numpy
            for b in range(len(labels)):
                self.grads[layer][labels[b]].append(grads[b])

    def sift(self):
        for layer in range(len(self.modules)):
            for label, grads in enumerate(self.grads[layer]):
                grads = np.asarray(grads)  # image_nums, val
                # print('grads.shape', grads.shape)

                # grads = np.maximum(grads, 0)  # - to 0
                grads = np.sum(grads, axis=0)  # sum image_nums

                mask = np.zeros(grads.shape)
                threshold = grads.mean()
                mask[np.where(grads > threshold)] = 1

                method_name = 'inputs_label{}_layer{}'.format(label, layer)  # io
                mask_path = os.path.join(self.result_path, 'grads_{}.npy'.format(method_name))
                np.save(mask_path, mask)

                # self.visualize(grads, mask, method_name)

    def visualize(self, grads, mask, method_name):
        grads = grads.reshape((256, -1))
        l_path = os.path.join(self.result_path, 'grads_{}.png'.format(method_name))
        image_util.view_grads(grads.transpose((1, 0)), grads.shape[0], grads.shape[1], l_path)

        mask = mask.reshape((256, -1))
        l_path = os.path.join(self.result_path, 'grads_{}_m.png'.format(method_name))
        image_util.view_grads(mask.transpose((1, 0)), mask.shape[0], mask.shape[1], l_path)


def main():
    data_name = 'cifar-10-2'
    model_name = 'simnet'
    checkpoint_name = model_name + '_01201934'
    # data_name = 'mini-imagenet-1'
    # model_name = 'alexnet'
    # checkpoint_name = model_name + '_11291613'
    # data_name = 'mini-imagenet-1'
    # model_name = 'vgg16'
    # checkpoint_name = model_name + '_11291535'

    model_path = os.path.join(config.outputs_model, checkpoint_name, 'checkpoint.pth')
    input_path = os.path.join(config.outputs_result, checkpoint_name, 'images_50')
    result_path = os.path.join(config.outputs_result, checkpoint_name, 'grads')

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
    modules = models.load_modules(model=model, model_layers=None)[0:-1]  # no first conv

    grad_sift = GradSift(modules=modules,
                         class_nums=cfg['model']['num_classes'],
                         result_path=result_path)

    data_loader = data_util.load_data(input_path)
    for i, samples in enumerate(data_loader):
        print('\r[{}/{}]'.format(i, len(data_loader)), end='', flush=True)
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        grad_sift(outputs, labels)

    grad_sift.sift()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    np.set_printoptions(threshold=np.inf)

    main()
