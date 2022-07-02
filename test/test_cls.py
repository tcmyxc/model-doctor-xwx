"""
此脚本只是为了验证模型的acc
"""
from collections import OrderedDict
import sys
sys.path.append('/nfs/xwx/model-doctor-xwx')

import torch
import models
import loaders
import argparse
import os
import time

from configs.config_util import get_cfg
from utils.time_util import print_time
from sklearn.metrics import classification_report
from modify_kernel.util.cfg_util import print_yml_cfg
from utils.args_util import print_args


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='cifar-10-lt-ir10')
parser.add_argument('--model_name', default='resnet32')
parser.add_argument('--data_loader_type', type=int, default='0', help='0 is default, 1 for cbs')
parser.add_argument('--model_path', default='/nfs/xwx/model-doctor-xwx/cifar10_imb01_stage2_mislas.pth.tar')


def main():
    args = parser.parse_args()
    print_args(args)

    # get cfg
    data_name    = args.data_name
    model_name   = args.model_name
    model_path   = args.model_path
    cfg_filename = "one_stage.yml"
    cfg = get_cfg(cfg_filename)[data_name]
    print_yml_cfg(cfg)

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-' * 42, '\n[Info] use device ', device)

    # data loader
    if args.data_loader_type == 0:
        # 常规数据加载器
        data_loaders, _ = loaders.load_data(data_name=data_name)
    elif args.data_loader_type == 1:
        # 类平衡采样
        data_loaders, _ = loaders.load_class_balanced_data(data_name=data_name)

    # model
    model = models.load_model(
        model_name=model_name, 
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"]
    )
    
    loaded_dict = torch.load(model_path)['state_dict_model']
    new_state_dict = OrderedDict()
    for k, v in loaded_dict.items():
        name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
        new_state_dict[name] = v #新字典的key值对应的value一一对应
        
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)

    begin_time = time.time()
    test(data_loaders["val"], model, device, args, cfg)
    print("Done!")
    print_time(time.time()-begin_time)


def test(dataloader, model, device, args, cfg):
    # 这里加入了 classification_report
    y_pred_list = []
    y_train_list = []
    
    size = len(dataloader.dataset)
    model.eval()
    correct = 0
    for batch, (X, y, _) in enumerate(dataloader):
        y_train_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)
        with torch.set_grad_enabled(True):
            pred, _ = model(X)

        y_pred_list.extend(pred.argmax(1).cpu().numpy())

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
        
    print(f"\nTest Error: Accuracy: {(100*correct):>0.2f}% \n")
    print(classification_report(y_train_list, y_pred_list, digits=4))



if __name__ == '__main__':
    main()