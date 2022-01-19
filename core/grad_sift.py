import sys
import os
import numpy as np
import torch
import json


sys.path.append('/nfs/xwx/model-doctor-xwx')
import loaders
import models
from configs import config
from utils import image_util
from core.grad_constraint import HookModule

# 高低响应通道的筛选，基本上就是取最大响应
class GradSift:
    def __init__(self, class_nums, grad_nums):
        self.class_nums = class_nums  # 数据集的类别总数
        self.grad_nums = grad_nums   # 每个数据集用多少个样本生成channel
        self.grads = None
        self.scores = torch.zeros((class_nums, grad_nums)) # 用于筛选每个类别对应的样本
        self.nums = torch.zeros(class_nums, dtype=torch.long) # 几个类，num[i]代表第i个样本的数量

    def __call__(self, outputs, labels, grads):
        # 计算每类预测正确不同图像的置信度和grads，
        # 并对置信度最大的grad_nums个样本其进行保存，每类图像预测正确的数目最多为grad_nums
        if self.grads is None:                   
            self.grads = torch.zeros((self.class_nums,
                                      self.grad_nums,
                                      grads.shape[1],  # C
                                      grads.shape[2],  # H
                                      grads.shape[3])) # W
            print(self.grads.shape)

        softmax = torch.softmax(outputs, dim=1)
        scores, predicts = torch.max(softmax, dim=1) # 通道维度上的最大值以及索引（也就是正确的标签）
        # 1、先往每个类的行向量里面放100个值
        # 2、找最小值
        # 3、继续读入下一个值，如果比最小的值大，则用该值替换最小值
        # 4、重复步骤2
        for i, label in enumerate(labels):
            if label == predicts[i]:
                # 如果预测正确
                if self.nums[label] == self.grad_nums:
                    # 如果该label采样达到了100次，以后的采样这么处理
                    # 每次都替换最小的值
                    score_min, index = torch.min(self.scores[label], dim=0)
                    if scores[i] > score_min:
                        # 如果后面的预测结果比其中的最小的预测结果大，用这个结果代替最小的预测结果
                        self.scores[label][index] = scores[i]
                        self.grads[label][index] = grads[i]
                else:
                    self.scores[label][self.nums[label]] = scores[i]
                    self.grads[label][self.nums[label]] = grads[i]
                    self.nums[label] += 1  # 第i类的数量加1

    def sum_channel(self, result_path, model_layer, epoch):
        # print(self.scores)
        # print(self.nums)
        flag = 1
        if flag == 1:
            # 原始模型医生
            grads = torch.abs(self.grads) # grads : (B, C, H, W)
        elif flag == 2:
            # 只用正梯度
            grads = torch.nn.ReLU()(self.grads)

        view_channel(grads, result_path, model_layer, epoch)
        # grads_pos = torch.nn.ReLU()(self.grads)
        # grads_neg = torch.nn.ReLU()(-self.grads)


def view_channel(grads, result_path, model_layer, epoch):
    """保存梯度并可视化"""
    # grads numpy
    # （1， 3， 4） = 一个类别中对应不同通道的grads求和, 保留 (Class, channel)
    grads_sum = torch.sum(grads, dim=(1, 3, 4)).detach().numpy()
    grads_path = os.path.join(
        result_path, 
        'channel_grads_{}_epoch{}.npy'.format(model_layer, epoch)
    )
    np.save(grads_path, grads_sum)

    # grads numpy view
    grads_path = os.path.join(
        result_path, 
        'channel_grads_{}_epoch{}.png'.format(model_layer, epoch)
    )
    image_util.view_grads(grads_sum, 512, 10, grads_path)

    # # grads numpy sort view
    # grads_sum_sort = -np.sort(-grads_sum, axis=1)
    # grads_path = os.path.join(result_path, 'channel_grads_{}_sort.png'.format(model_layer))
    # image_util.view_grads(grads_sum_sort, 512, 10, grads_path)
    #
    sift_channel(result_path, model_layer, epoch)


def sift_channel(result_path, model_layer, epoch, threshold=None):  # high response channel
    """根据梯度生成 channel mask，同时保存到文件"""
    grads_path = os.path.join(
        result_path, 
        'channel_grads_{}_epoch{}.npy'.format(model_layer, epoch)
    )
    channels_grads = np.load(grads_path)   # (Class, channel)

    if threshold is None:
        channels_threshold = channels_grads.mean(axis=1) # 对每个类对应的channel取平均，得到[Class, ]
    else:
        channels_threshold = -np.sort(-channels_grads, axis=1)[:, threshold]  # -sort从小到大, 得到[Class, ], 列选择threshold对应的列

    channels = np.ones(shape=channels_grads.shape)  # 先来个全1矩阵
    for c, t in enumerate(channels_threshold):
        # c为channels的行， t为channels_threshold的每一个值
        # 如果大于均值，则为1
        channels[c] = np.where(channels_grads[c] >= t, 1, 0)  # channels : [Class, channels]

    channel_path = os.path.join(
        result_path, 
        'channels_{}_epoch{}.npy'.format(model_layer, epoch)
    )
    np.save(channel_path, channels)  # 相当于将预测正确类 对信道响应大于平均值的信道置1，其余置0

    png_channel_path = os.path.join(
        result_path, 
        'channels_{}_epoch{}.png'.format(model_layer, epoch)
    )
    image_util.view_grads(channels, 512, 10, png_channel_path)

    # print(channels)
    # print(channels_threshold)


# ----------------------------------------
# test
# ----------------------------------------
def sift_grad(data_name, model_name, model_layers, model_path, result_path, epoch):
    # device
    device = torch.device('cuda:0')

    # config
    cfg = json.load(open('configs/config_trainer.json'))[data_name]

    # model（输入是3通道，最后输出类别是10）
    model = models.load_model(model_name=model_name,
                              in_channels=cfg['model']['in_channels'], # 输入图像的通道数
                              num_classes=cfg['model']['num_classes']) # 类别数
    # 加载预训练的模型参数，并设为推理模式
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    model.to(device)

    # data
    train_loader, _ = loaders.load_data(data_name=data_name, data_type='train')
    print(_)

    # grad
    module = HookModule(model=model, module=models.load_modules(model, model_name, model_layers)[0]) # 指定layer加入hook
    grad_sift = GradSift(class_nums=cfg['model']['num_classes'], grad_nums=100)  # 每个类别T个sample，对应grad_nums

    # forward
    for i, samples in enumerate(train_loader): # 对train_loader中所有数据进行预测
        print('\r[{}/{}]'.format(i, len(train_loader)), end='', flush=True)
        inputs, labels = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        
        activations = module.activations
        # 取正激活部分的索引
        act_idx = torch.ge(activations, torch.zeros_like(activations))
        # print("=" * 42)
        # print("\n==> activations\n", activations)
        grads = module.grads(
            outputs=-nll_loss, 
            inputs=activations, # 预测结果对于特定feature map计算梯度
            retain_graph=True, 
            create_graph=False)   # grads : [B, C, H, W]

        # print("\n==> nll_loss\n", nll_loss)
        # print("=" * 42)
        nll_loss.backward()  # to release graph

        # 计算梯度
        # print("\n==> grads\n", grads)
        flag = 1
        if flag == 1:
            # 原有的梯度
            pass
        elif flag == 2:
            # 正激活部分对应的梯度
            grads = grads * act_idx
        elif flag == 3:
            # 直接用激活代替梯度
            grads = activations
        elif flag == 4:
            # 用正激活代替梯度
            grads = torch.nn.ReLU()(activations)
        # print("\n==> grads\n", grads)
        
        # 调用 call 函数
        grad_sift(outputs=outputs, labels=labels, grads=grads)

    # 全部筛选完后对每个类别留下grad_nums的样本（针对某一层求的grads）
    print('\n', end='', flush=True)
    grad_sift.sum_channel(result_path, model_layers[0], epoch) # 加和


def main():
    # 数据集(dataset name)
    data_name = 'cifar-100-lt-ir100'
    # 模型名(model name)
    model_list = [
        # 'alexnetv3',
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
    model_name = model_list[0]
    model_layers = [-1]
    epoch = 0  # best weight

    # 模型路径(model path)
    model_path = os.path.join(
        config.model_pretrained, 
            config.model_pretrained, 
        config.model_pretrained, 
            config.model_pretrained, 
        config.model_pretrained, 
        f"{model_name}-{data_name}",
        "checkpoint.pth"
    )
    if not os.path.exists(model_path):
        print("-" * 79, "\n ERROR, the model path does not exist")
        return
  
    print("-" * 79, "\n model_path:", model_path)

    # 保存结果路径(the channel mask path)
    result_path = os.path.join(
        config.result_channels, 
            config.result_channels, 
        config.result_channels, 
            config.result_channels, 
        config.result_channels, 
        f"{model_name}-{data_name}"
    )

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    print("-" * 79, "\n result_path:", result_path)
    # return  # for test

    readme_file_path = os.path.join(result_path, "README.MD")
    with open(readme_file_path, 'w') as readme_file:
        readme_file.write("倒数第一层卷积层")
    

    sift_grad(data_name, model_name, model_layers, model_path, result_path, epoch)

    # 对不同 epoch 生成 channel mask 文件
    # for epoch in range(0, 201, 5):
    #     # 2021-12-25 modify
    #     # 预训练模型
    #     model_path = os.path.join(
    #         config.model_pretrained, 
    #         "resnet50-cifar-10-prune",
    #         f'checkpoint-{epoch}.pth'
    #     )
    #     print("model_path:", model_path)
    
    #     result_path = os.path.join(
    #         config.result_channels, 
    #         "resnet50-cifar-10-prune-ztd"
    #     )

    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)

    #     sift_grad(data_name, model_name, model_layers, model_path, result_path, epoch)

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    np.set_printoptions(threshold=np.inf)
    main()
# python core/grad_sift.py
