import torch
from core.grad_constraint import HookModule


class GradIntegral:
    def __init__(self, model, modules):
        print('- Grad Integral')

        self.modules = modules
        self.hooks = []

    def add_noise(self):
        for module in self.modules:
            hook = module.register_forward_hook(_modify_feature_map)
            self.hooks.append(hook)

    def remove_noise(self):
        for hook in self.hooks:
            hook.remove()
            self.hooks.clear()


# keep forward after modify
def _modify_feature_map(module, inputs, outputs):
    noise = torch.randn(outputs.shape).to(outputs.device)
    # noise = torch.normal(mean=0, std=3, size=outputs.shape).to(outputs.device)

    outputs += noise


def _test():
    import models
    from configs import config

    model = models.load_model('simnet')
    model_path = r'{}/{}/checkpoint.pth'.format(config.output_model, 'simnet_06171610')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
    model.eval()

    gi = GradIntegral(model=model, modules=[model.features.c3])
    module = HookModule(model=model, module=model.features.c3)

    inputs = torch.ones((4, 3, 224, 224))
    labels = torch.tensor([1, 1, 1, 1])

    print('-' * 10)
    gi.add_noise()
    outputs = model(inputs)
    print(outputs)

    print('-' * 10)
    gi.remove_noise()
    outputs = model(inputs)
    print(outputs)

    print('-' * 10)
    gi.add_noise()
    outputs = model(inputs)
    print(outputs)
    nll_loss = torch.nn.NLLLoss()(outputs, labels)
    grads = module.grads(outputs=-nll_loss, inputs=module.activations)
    print(grads)


if __name__ == '__main__':
    _test()

    # 9, 224, 224 -> 27, 112, 112 -> 81, 56, 56
    # noise = torch.randn(size=(4, 81, 56, 56))
    # print(noise)
    # noise = torch.normal(mean=0, std=3, size=(4, 81, 56, 56))
    # print(noise)
