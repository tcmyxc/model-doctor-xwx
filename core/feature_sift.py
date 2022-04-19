"""
特征图
"""
import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

import torch
import argparse
import numpy as np
import os
import models

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='cifar-10-lt-ir100')

class HookModule:
    """hook module"""

    def __init__(self, module):
        self.inputs = None
        self.activations = None

        module.register_forward_hook(self._hook_activations)

    def _hook_activations(self, module, inputs, outputs):
        self.inputs = inputs
        self.activations = outputs


class FeatureSift():

    def __init__(self, modules, class_nums, result_path) -> None:
        self.modules = modules
        self.result_path = result_path
        self.h_modules = [HookModule(module) for module in modules]
        
        self.features = [[[] for _ in range(class_nums)] for _ in range(len(modules))]
        
    def __call__(self, labels):
        for layer, h_module in enumerate(self.h_modules):
            output = h_module.activations.detach().cpu().numpy()  # batch_size*kernel_num*fsize*fsize
            for b in range(len(labels)):
                self.features[layer][labels[b]].append(output[b])

    def sift(self):
        # np.save(os.path.join(self.result_path, "features_all.npy"), np.array(self.features, dtype=object))
        for layer in range(len(self.modules)):
            for label, feature in enumerate(self.features[layer]):
                print(feature)
                root_path = os.path.join(self.result_path, str(layer), str(label))
                if not os.path.exists(root_path):
                    os.makedirs(root_path)

                feature_path = os.path.join(root_path, f'feature_label{label}_layer{layer}.npy')
                np.save(feature_path, feature)



def main():
    from modify_kernel.util.cfg_util import get_cfg
    import loaders
    args = parser.parse_args()
    print(f"\n[INFO] args: {args}")

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('\n[INFO] train on ', device)

    data_name = args.data_name
    cfg_filename = "cbs_refl.yml"
    cfg = get_cfg(cfg_filename)[data_name]

    model_name = cfg["model_name"]
    model_path = cfg["two_stage_model_path"]
    
    result_path = os.path.join(f"/nfs/xwx/model-doctor-xwx/output/result/{model_name}-{data_name}", 'features')

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # model
    model = models.load_model(model_name=model_name,
                              in_channels=cfg['model']['in_channels'],
                              num_classes=cfg['model']['num_classes'])
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    model.to(device)

    # modules
    modules = models.load_modules(model=model, model_name=model_name, model_layers=[29])  # no first conv
    print("\n modules:", modules)

    feature_sift = FeatureSift(modules=modules,
                         class_nums=cfg['model']['num_classes'],
                         result_path=result_path)

    data_loaders, _ = loaders.load_data(data_name=data_name)
    data_loader = data_loaders["train"]
    for i, samples in enumerate(data_loader):
        print('\r[{}/{}]'.format(i+1, len(data_loader)), end='', flush=True)
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        model(inputs)
        feature_sift(labels)
    print("\n")
    feature_sift.sift()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    np.set_printoptions(threshold=np.inf)

    main()
            

