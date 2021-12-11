import torch
from core.grad_constraint import HookModule


class GradIntegral:
    def __init__(self, model, modules):
        print('==> Grad Integral')

        self.modules = modules
        self.hooks = []

    def add_noise(self):
        for module in self.modules:
            hook = module.register_forward_hook(_modify_feature_map)
            self.hooks.append(hook)

    def remove_noise(self):
        for hook in self.hooks:
            hook.remove()  # 移除钩子

        self.hooks.clear()


# keep forward after modify，随机噪声
def _modify_feature_map(module, inputs, outputs):
    noise = torch.randn(outputs.shape).to(outputs.device)
    # noise = torch.normal(mean=0, std=3, size=outputs.shape).to(outputs.device)

    outputs += noise
