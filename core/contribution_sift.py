import sys

sys.path.append('.')

import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

import models
import loaders
from utils import image_util


def partial_conv(inp: torch.Tensor, conv: nn.Conv2d, o_h=None, o_w=None):
    kernel_size = conv.kernel_size
    dilation = conv.dilation
    padding = conv.padding
    stride = conv.stride
    weight = conv.weight  # O I K K
    bias = conv.bias  # O

    wei_res = weight.view(weight.size(0), weight.size(1), -1).permute((1, 2, 0))  # I K*K O
    inp_unf = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)(inp)  # B K*K N
    inp_unf = inp_unf.view(inp.size(0), inp.size(1), wei_res.size(1), o_h, o_w)  # B I K*K H_O W_O
    out = torch.einsum('ijkmn,jkl->iljmn', inp_unf, wei_res)  # B O I H W
    # out = out.sum(2)
    # bias = bias.unsqueeze(1).unsqueeze(2).expand((out.size(1), out.size(2), out.size(3)))  # O H W
    # out = out + bias
    return out


def partial_linear(inp: torch.Tensor, linear: nn.Linear):
    # inp: B I
    weight = linear.weight  # [O I]
    bias = linear.bias  # O

    B = inp.shape[0]
    I = weight.shape[1]
    O = weight.shape[0]

    inp = inp.unsqueeze(1).expand(B, O, I)
    weight = weight.unsqueeze(0).expand(B, O, I)
    out = inp * weight
    # out = out.sum(2)
    # out = out + bias
    return out  # [B O I]


class HookModule:
    def __init__(self, module, value_type):
        self.module = module
        self.value_type = value_type
        module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        self.outputs = outputs
        if isinstance(module, nn.Conv2d):
            partial_outputs = partial_conv(inputs[0], module, outputs.size(2), outputs.size(3))  # [b, o, i, h, w]
            if self.value_type == '+':
                partial_outputs = torch.relu(torch.sum(partial_outputs, dim=(3, 4)))  # [b, o, i]
            self.p_outputs = partial_outputs
        if isinstance(module, nn.Linear):
            partial_outputs = partial_linear(inputs[0], module)  # [b, o, i]
            if self.value_type == '+':
                partial_outputs = torch.relu(partial_outputs)  # [b, o, i]
            self.p_outputs = partial_outputs


class ActivationSift:
    def __init__(self, modules, num_classes, value_type):
        self.modules = [HookModule(module, value_type) for module in modules]
        # self.values = [[[] for _ in range(num_classes)] for _ in range(len(modules))]
        # [num_modules, num_classes, channels]
        self.values = [[0 for _ in range(num_classes)] for _ in range(len(modules))]
        # [num_modules, num_classes, num_images, channels]
        self.value_type = value_type

    def __call__(self, outputs, labels):
        for layer, module in enumerate(self.modules):
            values = None
            if isinstance(module.module, nn.Conv2d):
                values = module.p_outputs
            elif isinstance(module.module, nn.Linear):
                values = module.p_outputs

            values = values.detach().cpu().numpy()

            for b in range(len(labels)):
                # self.values[layer][labels[b]].append(values[b])
                self.values[layer][labels[b]] += values[b]

    def _normalization(self, data, axis=None, bot=False):
        assert axis in [None, 0, 1]
        _max = np.max(data, axis=axis)
        if bot:
            _min = np.zeros(_max.shape)
        else:
            _min = np.min(data, axis=axis)
        _range = _max - _min
        if axis == 1:
            _norm = ((data.T - _min) / (_range + 1e-5)).T
        else:
            _norm = (data - _min) / (_range + 1e-5)
        return _norm

    def sift(self, result_path):
        for layer, values in enumerate(self.values):  # [num_modules, num_classes, num_images, o, i]
            values = np.asarray(self.values[layer])  # [num_classes, num_images, o, i]
            # values = np.sum(values, axis=1)  # [num_classes, o, i]
            print(values.shape)

            if isinstance(self.modules[layer].module, nn.Conv2d):  # [num_classes, o, i]
                print('*******')
                alpha = 0.3
                beta = 0.2
                masks = []
                for data in values:  # [o, i]
                    data = self._normalization(data, axis=1)  # [o, i]
                    mask = np.zeros(data.shape)
                    mask[np.where(data > alpha)] = 1  # [o, i]
                    mask = np.sum(mask, axis=0)  # [i]
                    masks.append(mask)  # [num_classes,  i]
                masks = np.asarray(masks)  # [num_classes,  i]

                masks = self._normalization(masks, axis=1, bot=True)
                mask = np.zeros(masks.shape)
                mask[np.where(masks > beta)] = 1

                mask_path = os.path.join(result_path, '{}_layer{}.npy'.format(self.value_type, layer))
                np.save(mask_path, mask)

            if isinstance(self.modules[layer].module, nn.Linear):  # [num_classes, o, i]
                print('-------')
                alpha = 0.4
                beta = 0.3
                masks = []
                for data in values:  # [o, i]
                    # print(data.shape)
                    data = self._normalization(data, axis=1)
                    mask = np.zeros(data.shape)
                    mask[np.where(data > alpha)] = 1  # [o, i]
                    mask = np.sum(mask, axis=0)  # [i]
                    masks.append(mask)  # [num_classes,  i]
                masks = np.asarray(masks)  # [num_classes,  i]

                masks = self._normalization(masks, axis=1, bot=True)  # [num_classes,  i]
                mask = np.zeros(masks.shape)
                mask[np.where(masks > beta)] = 1

                mask_path = os.path.join(result_path, '{}_layer{}.npy'.format(self.value_type, layer))
                np.save(mask_path, mask)

                # the connection between conv and linear
                if isinstance(self.modules[layer + 1].module, nn.Conv2d):
                    values_conv = np.asarray(self.values[layer + 1])  # [num_classes, o, i]
                    p = int(values.shape[2] / values_conv.shape[1])
                    masks = np.reshape(masks, (values_conv.shape[0], values_conv.shape[1], p))
                    masks = np.sum(masks, axis=2)  # [num_classes, o] of conv

                    masks = self._normalization(masks, axis=1, bot=True)  # [num_classes,  i]
                    mask = np.zeros(masks.shape)
                    mask[np.where(masks > beta)] = 1

                    mask_path = os.path.join(result_path, '{}_layer{}_o.npy'.format(self.value_type, layer + 1))
                    np.save(mask_path, mask)

    def visualize_global_by_labels(self, result_path, layers):
        for layer in layers:
            # [num_modules, num_classes, num_images, o, i]
            values = np.asarray(self.values[layer])  # [num_classes, num_images, o, i]
            # values = np.sum(values, axis=1)  # [num_classes, o, i]

            # -------------------------------------------------
            alpha = 0.3
            masks = []
            for data in values:  # [o, i]
                data = self._normalization(data, axis=1)  # [o, i]
                mask = np.zeros(data.shape)
                mask[np.where(data > alpha)] = 1  # [o, i]
                mask = np.sum(mask, axis=0)  # [i]
                masks.append(mask)  # [num_classes,  i]
            values = np.asarray(masks)  # [num_classes,  i]
            # -------------------------------------------------

            # values = np.sum(values, axis=1)  # [num_classes, i]

            result_name = '{}_norm_global_layer{}.png'.format(self.value_type, layer)
            values_path = os.path.join(result_path, result_name)
            image_util.heatmap(values, values_path, fig_w=256, fig_h=40, annot=False)

    def visualize_global_by_images(self, result_path, layers, labels):
        for layer in layers:
            for label in labels:
                # [num_modules, num_classes, num_images, o, i]
                values = np.asarray(self.values[layer])  # [num_classes, num_images, o, i]
                values = values[label, 0:10]  # [num_images, o, i]
                values = np.sum(values, axis=1)  # [num_images, i]

                result_name = '{}_global_layer{}_label{}.png'.format(self.value_type, layer, label)
                values_path = os.path.join(result_path, result_name)
                image_util.heatmap(values, values_path, fig_w=256, fig_h=40, annot=False)

    def visualize_local_by_labels(self, result_path, layers, kernels):
        for i, values in enumerate(self.values):
            for kernel in kernels:
                # [num_modules, num_classes, num_images, o, i]
                values = np.asarray(values)  # [num_classes, num_images, o, i]
                # values = np.sum(values, axis=1)  # [num_classes, o, i]
                values = values[:, kernel]  # [num_classes, i]

                result_name = '{}_local_layer{}_kernel{}.png'.format(self.value_type, layers[i], kernel)
                values_path = os.path.join(result_path, result_name)
                image_util.heatmap(values, values_path, fig_w=256, fig_h=40, annot=False)

    def visualize_local_by_images(self, result_path, layers, labels, kernels):
        for layer in layers:
            for label in labels:
                for kernel in kernels:
                    # [num_modules, num_classes, num_images, o, i]
                    values = np.asarray(self.values[layer])  # [num_classes, num_images, o, i]
                    values = values[label, 0:10, kernel]  # [num_images, i]

                    result_name = '{}_local_layer{}_label{}_kernel{}.png'.format(self.value_type, layer, label, kernel)
                    values_path = os.path.join(result_path, result_name)
                    image_util.heatmap(values, values_path, fig_w=256, fig_h=40, annot=False)

    def visualize_aggregation_by_labels(self, result_path, layers, indexs):
        for layer in layers:
            for index in indexs:
                # [num_modules, num_classes, num_images, o, i]
                values = np.asarray(self.values[layer])  # [num_classes, num_images, o, i]
                values = np.sum(values, axis=1)  # [num_classes, o, i]
                values = values[0:10, :, index]  # [num_classes, o]

                result_name = '{}_agg_layer{}_index{}.png'.format(self.value_type, layer, index)
                values_path = os.path.join(result_path, result_name)
                image_util.heatmap(values, values_path, fig_w=256, fig_h=40, annot=True)

    def visualize_aggregation_by_images(self, result_path, layers, labels, indexs):
        for layer in layers:
            for label in labels:
                for index in indexs:
                    # [num_modules, num_classes, num_images, o, i]
                    values = np.asarray(self.values[layer])  # [num_classes, num_images, o, i]
                    values = values[label, 0:10, :, index]  # [num_images, o]

                    result_name = '{}_agg_layer{}_label{}_index{}.png'.format(self.value_type, layer, label, index)
                    values_path = os.path.join(result_path, result_name)
                    image_util.heatmap(values, values_path, annot=True)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--loader_name', default='', type=str, help='loader name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_path', default='', type=str, help='data path')
    parser.add_argument('--activation_path', default='', type=str, help='pattern path')
    parser.add_argument('--device_index', default='0', type=str, help='device index')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.activation_path):
        os.makedirs(args.activation_path)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('DATA PATH:', args.data_path)
    print('RESULT PATH:', args.activation_path)
    print('-' * 50)

    # ----------------------------------------
    # model/data configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    # model = torch.load(args.model_path)
    model.to(device)
    model.eval()

    # data_loader = loaders.load_data(loader_name=args.loader_name, data_path=args.data_path)

    modules = models.load_modules(model=model)

    activation_sift = ActivationSift(modules=modules, num_classes=args.num_classes, value_type='+')

    # ----------------------------------------
    # forward
    # ----------------------------------------
    for i, samples in enumerate(tqdm(data_loader)):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)

        activation_sift(outputs, labels)

    activation_sift.sift(result_path=args.activation_path)

    # activation_sift.visualize_global_by_labels(result_path=args.activation_path,
    #                                            layers=[2])

    # activation_sift.visualize_global_by_images(result_path=args.activation_path,
    #                                            layers=[3, 13],
    #                                            labels=[1])
    #
    # activation_sift.visualize_local_by_labels(result_path=args.activation_path,
    #                                           layers=[3, 13],
    #                                           kernels=[1])
    #
    # activation_sift.visualize_local_by_images(result_path=args.activation_path,
    #                                           layers=[3, 13],
    #                                           labels=[1],
    #                                           kernels=[1])

    # activation_sift.visualize_aggregation_by_labels(result_path=args.activation_path,
    #                                                 layers=[3, 13],
    #                                                 indexs=[1])
    #
    # activation_sift.visualize_aggregation_by_images(result_path=args.activation_path,
    #                                                 layers=[3, 13],
    #                                                 labels=[1],
    #                                                 indexs=[1])


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    main()
