import torch

class GradHookModule:
    """hook module"""

    def __init__(self, model, module, class_nums):
        self.model = model
        self.inputs = None
        self.activations = None

        module.register_forward_hook(self._hook_activations)

    def _hook_activations(self, module, inputs, outputs):
        self.inputs = inputs
        self.activations = outputs

    def grads(self, outputs, inputs, targets, retain_graph=True, create_graph=True):
        grads = torch.autograd.grad(outputs=outputs,
                                    inputs=inputs,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)[0].detach()  # batch_size*kernel_size*H*W
        self.model.zero_grad()

        grads = torch.relu(grads)
        grads = torch.sum(grads, dim=(0, 2, 3))
        
        kernel_percent = grads / grads.sum()

        return kernel_percent