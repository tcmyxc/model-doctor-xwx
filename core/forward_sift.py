import sys
# 代码路径，让python解释器可以找到路径
sys.path.append('/home/xwx/model-doctor-xwx/')
import os
import numpy as np
import torch
from torch import nn
import json

import models
from configs import config_path
from utils import image_util
from utils import data_util


class HookModule:
    def __init__(self, module):
        self.module = module
        self.inputs = None
        self.activations = None

        module.register_forward_hook(self._hook_activations)

    def grads(self, outputs, inputs=None, retain_graph=True, create_graph=False):
        if inputs is None:
            inputs = self.activations

        grads = torch.autograd.grad(outputs=outputs,
                                    inputs=inputs,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)[0]
        return grads

    def _hook_activations(self, module, inputs, outputs):
        self.inputs = inputs[0]
        # print('--', self.inputs.shape)
        self.activations = outputs


def split_conv(module: nn.Conv2d, inputs: torch.Tensor, is_sum=False):
    outputs = []
    for idx in range(inputs.shape[1]):  # in_channels
        outputs_in = torch.conv2d(input=inputs[:, idx].unsqueeze(1),
                                  weight=module.weight[:, idx].unsqueeze(1),
                                  bias=None,
                                  stride=module.stride,
                                  padding=module.padding,
                                  dilation=module.dilation,
                                  groups=module.groups)
        outputs.append(outputs_in)

    outputs = torch.stack(outputs, dim=2)

    if is_sum:
        outputs = torch.sum(outputs, dim=2)
        bias = torch.unsqueeze(module.bias, dim=0)
        bias = torch.unsqueeze(bias, dim=2)
        bias = torch.unsqueeze(bias, dim=3)
        bias = bias.expand(outputs.shape[0],
                           outputs.shape[1],
                           outputs.shape[2],
                           outputs.shape[3])
        outputs = outputs + bias
        return outputs
    else:
        return outputs


class KernelSift:
    def __init__(self, modules, class_nums, result_path):
        self.modules = []
        self.relations = [[None for _ in range(len(modules))] for _ in range(class_nums)]
        # [class_nums, module_nums:relation[c,c-1]]
        print(self.relations)

        self.result_path = result_path

        for module in modules:
            self.modules.append(HookModule(module))

    def __call__(self, labels):
        self.relation(labels)

    def relation(self, labels):
        for layer, module in enumerate(self.modules):

            outputs = split_conv(module.module, module.inputs)
            relations = torch.sum(torch.relu(-outputs), dim=(3, 4))  # TODO +-
            # relations = torch.sigmoid(relations, dim=1)
            relations = relations.detach().cpu().numpy()

            for b in range(len(labels)):
                relation = relations[b]
                if self.relations[labels[b]][layer] is None:
                    self.relations[labels[b]][layer] = relation
                else:
                    self.relations[labels[b]][layer] += relation

    def sift(self, label):
        # TODO output to last layer, bias
        channel_path = os.path.join(config_path.outputs_result,
                                    'alexnet_11291613',
                                    'channels',
                                    'channels_{}_{}.npy')
        bias = np.load(channel_path.format('g_+', 0))[label]

        mask_path = os.path.join(self.result_path, 'bias_{}_l{}_c{}.npy'.format('test-', 0, label))
        np.save(mask_path, bias)
        print(bias)

        # kernels = np.asarray([1])
        for layer, relations in enumerate(self.relations[label]):
            print('-' * 80)
            print('relations.shape', relations.shape)

            threshold = np.expand_dims(relations.mean(axis=0), axis=0)  # [[.],[.]]: c kernel num
            print('threshold.shape(grad)', threshold.shape)
            # print(threshold)
            mask = np.zeros(relations.shape)
            mask[np.where(relations > threshold)] = 1  # sift by grad threshold
            mask[np.where(bias != 1)] = 0  # sift by last layer bias (whether is used)

            # -----weight current
            # weight = mask
            # weight = np.expand_dims(weight, axis=2)
            # weight = np.repeat(weight, self.modules[layer].module.weight.shape[2], axis=2)
            # weight = np.expand_dims(weight, axis=3)
            # weight = np.repeat(weight, self.modules[layer].module.weight.shape[2], axis=3)
            print('weight shape', mask.shape)
            print(mask)
            mask_path = os.path.join(self.result_path, 'weight_{}_l{}_c{}.npy'.format('test-', layer, label))
            np.save(mask_path, mask)

            # -----bias previous
            mask = mask.sum(axis=0)  # correlation times of each convolution kernel ([., .])
            mask[np.where(mask > 1)] = 1  # If previous kernel is used, the position is 1
            print('bias shape', mask.shape)
            print(mask)
            mask_path = os.path.join(self.result_path, 'bias_{}_l{}_c{}.npy'.format('test-', layer + 1, label))
            np.save(mask_path, mask)
            bias = mask

            # relations_path = os.path.join(self.result_path, 'relations_{}_l{}_c{}.png'.format('-50', layer, 0))
            # image_util.view_grads(relations.transpose((1, 0)), relations.shape[0], relations.shape[1], relations_path)

    def visualize(self, relations, layer):
        pass


# ----------------------------------------
# test
# ----------------------------------------
def sift_kernel(data_name, model_name, model_layers, model_path, result_path, input_path):
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
    modules = models.load_modules(model=model,
                                  model_layers=model_layers)

    kernel_sift = KernelSift(modules=modules,
                             class_nums=cfg['model']['num_classes'],
                             result_path=result_path)

    data_path = os.path.split(input_path)[0]
    print(data_path)
    data_loader = data_util.load_data(data_path)
    for i, samples in enumerate(data_loader):
        print('\r[{}/{}]'.format(i, len(data_loader)), end='', flush=True)
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        kernel_sift(labels)

    kernel_sift.sift(label=0)


def main():
    # data_name = 'cifar-10-1'
    # model_name = 'simnet'
    # checkpoint_name = model_name + '_11041002'

    data_name = 'mini-imagenet-1'
    model_name = 'alexnet'
    checkpoint_name = model_name + '_11291613'

    model_layers = None  # default all, from end to beginning
    model_path = os.path.join(config_path.outputs_model, checkpoint_name, 'checkpoint.pth')

    img_class = ''
    img_name = 'test'
    input_path = os.path.join(config_path.outputs_result, checkpoint_name, 'images_50', img_class, img_name)
    result_path = os.path.join(config_path.outputs_result, checkpoint_name, 'activations')

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    sift_kernel(data_name, model_name, model_layers, model_path, result_path, input_path)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    np.set_printoptions(threshold=np.inf)

    main()
