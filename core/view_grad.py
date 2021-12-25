import sys

# 代码路径，让python解释器可以找到路径
sys.path.append('/home/xwx/model-doctor-xwx/')
import torch
import json

import models
from configs import config
import loaders

import matplotlib

matplotlib.use('AGG')
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


def sift_grad(data_name, model_name, model_layers, model_path):
    # device
    device = torch.device('cuda:0')
    # config
    cfg = json.load(open('configs/config_trainer.json'))[data_name]
    # model（输入是3通道，最后输出类别是10）
    model = models.load_model(model_name=model_name,
                              in_channels=cfg['model']['in_channels'],  # 输入图像的通道数
                              num_classes=cfg['model']['num_classes'])  # 类别数
    # 加载预训练的模型参数，并设为推理模式
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    model.to(device)

    # data
    train_loader, _ = loaders.load_data(data_name=data_name, data_type='test')

    # modules
    modules = models.load_modules(model, model_name, model_layers)
    new_modules = []
    for _, module in enumerate(modules):
        new_modules.append(HookModule(module))

    print("modules length:", len(new_modules))


    # forward
    for i, samples in enumerate(train_loader):  # 对train_loader中所有数据进行预测
        print('\r[{}/{}]'.format(i, len(train_loader)), end='', flush=True)
        inputs, labels, _ = samples
        print("img_name", _, "lable", labels)
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        nll_loss.backward()  # to release graph
        break

    # 训练结束，可视化结果
    for layer, module in enumerate(new_modules):
        activations = module.activations.cpu().detach()[0].numpy()  # 只取第一张图片
        plt.tight_layout()
        fig, axs = plt.subplots(4, 4, figsize=(32, 32))
        for idx, activation in enumerate(activations[:16]):
            ax = axs[idx // 4, idx % 4]
            # ax.axis("off")
            sns.heatmap(activation, ax=ax, square=True)
        plt.savefig(f"img/act-layer{layer}.png")
        plt.close(fig)



def main(model_name, data_name):
    model_layers = None  # 模型导数第一层，一般是全连接层

    # 预训练模型
    model_path = os.path.join(
        config.model_pretrained,
        "resnet50-20211208-101731",
        'checkpoint.pth'
    )
    print("model_path", model_path)

    sift_grad(data_name, model_name, model_layers, model_path)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    np.set_printoptions(threshold=np.inf)
    data_name = 'stl-10'
    model_list = [
        # 'alexnet',
        # 'vgg16',
        'resnet50',
        # 'senet34',
        # 'wideresnet28',
        # 'resnext50',
        # 'densenet121',
        # 'simplenetv1',
        # 'efficientnetv2s',
        # 'googlenet',
        # 'xception',
        # 'mobilenetv2',
        # 'inceptionv3',
        # 'shufflenetv2',
        # 'squeezenet',
        # 'mnasnet'
    ]
    for model_name in model_list:
        main(model_name, data_name)
