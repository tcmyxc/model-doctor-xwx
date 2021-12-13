import sys

# sys.path.append('/home/hjc/classification/')  # 137/208/111
sys.path.append('/disk2/hjc/classification/')  # 210
# sys.path.append('/new_disk_1/disk1/hjc/classification/')  # 205

import os
import torch
from torch import optim
from torch import nn

import models
import loaders

from configs import config
from trainers.cls_trainer import ClsTrainer


def main():
    # ----------------------------------------
    # initial
    # ----------------------------------------

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Train device on', device)
    # data
    train_loader, train_size = loaders.load_coco_images('train')
    test_loader, test_size = loaders.load_coco_images('val')
    data_loaders = {'train': train_loader, 'val': test_loader}
    dataset_sizes = {'train': train_size, 'val': test_size}
    print(dataset_sizes)
    # model
    model = models.load_model('simnet')

    # ----------------------------------------
    # loss function/optimizer/lr scheduler
    # ----------------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=1e-02, momentum=0.9)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # ----------------------------------------
    # train
    # ----------------------------------------
    model_path = os.path.join(config.output_result, 'simnet_p_06180950')
    checkpoint_path = os.path.join(config.output_result, 'model.pth')
    trainer = ClsTrainer(model=model,
                         criterion=criterion,
                         optimizer=optimizer,
                         scheduler=lr_scheduler,
                         data_loaders=data_loaders,
                         dataset_sizes=dataset_sizes,
                         device=device,
                         num_epochs=50,
                         result_path=model_path,
                         model_path=None,
                         num_classes=12)

    # trainer.train()
    trainer.check(model_path=checkpoint_path)


if __name__ == '__main__':
    main()
