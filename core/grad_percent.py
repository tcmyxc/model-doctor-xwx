import os
import torch

import numpy as np

class HookModule:
    def __init__(self, model, module):
        self.model = model
        self.activations = None
        self.inputs = None

        module.register_forward_hook(self._hook_activations)

    def _hook_activations(self, module, inputs, outputs):
        self.activations = outputs
        self.inputs = inputs

    def grads(self, outputs, inputs, retain_graph=True, create_graph=True):
        grads = torch.autograd.grad(outputs=outputs,
                                    inputs=inputs,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)[0]
        self.model.zero_grad()

        return grads


class KernelGrad:
    """modify kernel grad by class's weight"""
    def __init__(self, model, modules, kernel_percent_path):
        self.modules = []
        self.kernel_percents = []
        self.kernel_grads = []

        for module in modules:
            self.modules.append(HookModule(model=model, module=module))

        for layer in range(len(self.modules)):
            kernel_percent = np.load(os.path.join(kernel_percent_path, f"grads_percent_inputs_layer{layer}.npy"))
            self.kernel_percents.append(kernel_percent)

    
    def cal_kernel_grad(self, logits, targets):
        nll_loss = torch.nn.NLLLoss()(logits, targets)

        targets = targets.view(-1, 1)  # 多加一个维度，为使用 gather 函数做准备
        for i, module in enumerate(self.modules):
            activations = module.activations
            grads=module.grads(outputs=-nll_loss, inputs=activations)
            grad_i = grads.gather(1, targets)  # 每个类对应的梯度

