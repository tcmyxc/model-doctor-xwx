import numpy as np
import torch
from torchvision import transforms


class HookModule:
    """hook，对中间层的输入输出进行记录"""

    def __init__(self, model, module):
        """给model的某一层加钩子"""
        self.model = model
        self.activations = None

        module.register_forward_hook(self._hook_activations)

    # 钩子函数原型：hook(module, input, output) -> None or modified output
    def _hook_activations(self, module, inputs, outputs):
        """记录某一层layer输出的feature map"""
        self.activations = outputs

    def grads(self, outputs, inputs, retain_graph=True, create_graph=True):
        """默认保留梯度图，同时构建导数图（供计算高阶导数使用）"""
        grads = torch.autograd.grad(outputs=outputs,
                                    inputs=inputs,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)[0]
        self.model.zero_grad()

        return grads


def _loss_channel(channels, grads, labels, is_high=True):
    flag = 1
    if flag == 1:
        grads = torch.abs(grads)
    elif flag == 2:
        grads = torch.nn.ReLU()(grads)  # 只用正梯度
        
    channel_grads = torch.sum(grads, dim=(2, 3))  # [batch_size, channels] 梯度加和后得到 [Batch_size，Channel]

    loss = 0
    # # 正梯度按生成的channel mask约束, 负梯度全部约束(the result is not good)
    # grads_pos = torch.nn.ReLU()(grads)
    # grads_neg = torch.nn.ReLU()(-grads)
    # pos_channel_grads = torch.sum(grads_pos, dim=(2, 3))
    # neg_channel_grads = torch.sum(grads_neg)
    # # 负梯度直接约束
    # loss += neg_channel_grads
    # # 正梯度按生成的channel mask约束
    # channel_grads = pos_channel_grads
    
    if is_high:
        for i, l in enumerate(labels):  # b代表batch中不同的样本， l代表labels中的类别
            loss += (channel_grads[i] * channels[l]).sum()  # 错误类别的高响应
    else:
        for b, l in enumerate(labels):
            loss += (channel_grads[b] * (1 - channels[l])).sum()  # 正确类别的低响应
    loss = loss / labels.size(0)
    return loss
    # 总结为： 1. 对正确类别响应高的信道，对错误类别响应应该尽量低
    #         2. 对正确类别响应低的信道， 对于正确类别的低响应应该更低


class GradConstraint:

    def __init__(self, model, modules, channel_paths):
        """

        :param model: 模型名
        :param modules: 模型的某一层
        :param channel_paths: channel path
        """
        print('- Grad Constraint')
        self.modules = []
        self.channels = []

        for module in modules:
            # 给某一层增加 hook 以记录中间值
            self.modules.append(HookModule(model=model, module=module))
        for channel_path in channel_paths:
            # 从指定文件加载 channel 信息
            self.channels.append(torch.from_numpy(np.load(channel_path)).cuda())

    def loss_channel(self, outputs, labels):
        """
        channel lossxs
        :param outputs: 输出结果
        :param labels: 真实标签
        :return:
        """
        # torch.argsort: 将x中的元素从小到大排列，提取其在原始位置上的index(索引)作为结果返回
        probs = torch.argsort(-outputs, dim=1)  # 提取预测结果从大到小的标签
        labels_ = []  # 自定义的 label，用来计算自定义的 loss
        for i in range(labels.size(0)):
            if probs[i][0] == labels[i]:
                labels_.append(probs[i][1])  # TP rank2
            else:
                labels_.append(probs[i][0])  # FP rank1
        labels_ = torch.tensor(labels_).cuda()
        # NLL Loss: y => y = softmax(y) => y=log(y) => sum(|y|)/len(y)
        nll_loss_ = torch.nn.NLLLoss()(outputs, labels_)
        nll_loss = torch.nn.NLLLoss()(outputs, labels)

        loss = 0
        for i, module in enumerate(self.modules):
            # high response channel loss
            # 12月12日晚，使用正激活计算梯度
            # activations = torch.nn.ReLU()(module.activations)
            activations = module.activations
            # 取大于0的索引
            act_idx = torch.ge(activations, torch.zeros_like(activations))
            loss += _loss_channel(
                channels=self.channels[i],
                grads=module.grads(outputs=-nll_loss_, inputs= activations),
                labels=labels_,
                is_high=True
            )

            # low response channel loss
            loss += _loss_channel(
                channels=self.channels[i],
                grads=module.grads(outputs=-nll_loss, inputs=activations),
                labels=labels,
                is_high=False
            )

        return loss
    
    def loss_spatial(self, outputs, labels, masks):
        """空间损失"""
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        grads = self.modules[0].grads(outputs=-nll_loss, # grads : (B, C, W, H)
                                      inputs=self.modules[0].activations)
        masks = transforms.Resize((grads.shape[2], grads.shape[3]))(masks) # 将mask的size转化成与grads相同
        masks_bg = 1 - masks # 标注区域转化为除去目标的背景区域
        grads_bg = torch.abs(masks_bg * grads)

        loss = grads_bg.sum()
        return loss

