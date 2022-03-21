import torch

class HookModule:
    """hook """

    def __init__(self, model, module):
        self.model = model
        self.activations = None

        module.register_forward_hook(self._hook_activations)

    # ：hook(module, input, output) -> None or modified output
    def _hook_activations(self, module, inputs, outputs):
        self.activations = outputs

    def grads(self, outputs, inputs, retain_graph=True, create_graph=True):
        grads = torch.autograd.grad(outputs=outputs,
                                    inputs=inputs,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)[0]
        self.model.zero_grad()

        return grads