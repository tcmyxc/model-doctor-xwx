import argparse
import os
import random
import shutil
import time
import warnings
import json
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

PHASE_TRAIN = "train"
PHASE_EVAL = "val"


def train(self):
    lr_list = []  # 记录学习率变化
    since = time.time()
    original_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

    # ----------------------------------------
    # Each epochs.
    # ----------------------------------------
    for epoch in range(self.num_epochs):
        print('\nEpoch {}/{}'.format(epoch+1, self.num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                self.model.train()
                self.gi.add_noise()
            else:
                self.model.eval()
                self.gi.remove_noise()

            top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
            top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
            losses = AverageMeter('Loss', ':.4e', Summary.NONE)
            progress = ProgressMeter(
                len(self.data_loaders[phase]),
                [losses, top1, top5],
                prefix=f'{phase}: ')

            running_loss_cls = 1e-5
            running_loss_gc = 1e-5
            running_corrects = 1e-5
            class_correct = list(1e-5 for i in range(self.num_classes))
            class_total = list(1e-5 for i in range(self.num_classes))

            # ----------------------------------------
            # Iterate over data.
            # ----------------------------------------
            for i, samples in enumerate(self.data_loaders[phase]):
                inputs, labels = samples
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if phase == 'train':
                    # masks = masks.to(self.device)
                    pass

                self.optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss_cls = self.criterion(outputs, labels)
                    loss_spatial = torch.tensor(0)
                    loss_channel = torch.tensor(0)

                    loss_channel = self.gc.loss_channel(
                        outputs=outputs, labels=labels)
                    if phase == 'train':
                        # loss_spatial = self.gc.loss_spatial(outputs=outputs, labels=labels, masks=masks)
                        pass

                    loss_channel = loss_channel * 10
                    loss_gc = loss_channel + loss_spatial * 10
                    loss = loss_cls + loss_gc
                    if i % 10 == 0:
                        print('\r[{}/{}] loss_cls:{:.4f} loss_spatial:{:.4f} loss_channel:{:.4f}'
                              .format(i, len(self.data_loaders[phase]), loss_cls, loss_spatial, loss_channel),
                              end='', flush=True)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                    losses.update(loss.item(), inputs.size(0))
                    top1.update(acc1[0], inputs.size(0))
                    top5.update(acc5[0], inputs.size(0))

                    # backward
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                # ----------------------------------------
                # Calculate the loss and acc of each epoch
                # ----------------------------------------
                running_loss_cls += loss_cls.item() * inputs.size(0)
                running_loss_gc += loss_gc.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                correct_flag = (preds == labels.data)
                for j in range(len(inputs)):
                    label = labels[j]
                    class_correct[label] += correct_flag[j].item()
                    class_total[label] += 1
                
                if i % 100 == 0:
                    progress.display(i)

            # print("\n")
            # for i in range(self.num_classes):
            #     class_acc = 100 * class_correct[i] / class_total[i]
            #     # print('Accuracy of %2d : %2d %%' % (i, class_acc))
            #     print(f"acc of {i:2d} : {class_acc:.2f}%")
            epoch_loss_cls = running_loss_cls / self.dataset_sizes[phase]
            epoch_loss_gc = running_loss_gc / self.dataset_sizes[phase]
            epoch_acc = running_corrects / self.dataset_sizes[phase]
            print('\n[{}] Loss CLS: {:.4f} Loss GC: {:.4f} Acc: {:.4f}'
                  .format(phase, epoch_loss_cls, epoch_loss_gc, epoch_acc))
            print(f"\n[{phase}] acc1 is {top1.avg:.4f}, acc5 is {top5.avg:.4f}")
            progress.display_summary()

            # ----------------------------------------
            # Save epoch data
            # ----------------------------------------
            is_best = self.history.update(phase=phase,
                                          acc=epoch_acc,
                                          loss_cls=epoch_loss_cls,
                                          loss_gc=epoch_loss_gc)
            if is_best:
                print('- Update best weights')
                if not os.path.exists(self.result_path):
                    os.makedirs(self.result_path)
                state = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best': self.history.best,
                    'history': self.history.history,
                    "acc": f"{epoch_acc}"
                }
                path = os.path.join(self.result_path, 'checkpoint.pth')
                torch.save(state, path)

            if phase == 'val':
                self.history.draw()
                adjust_learning_rate(self.optimizer, epoch, original_lr)
                cur_lr = float(self.optimizer.state_dict()
                               ['param_groups'][0]['lr'])
                lr_list.append(cur_lr)
                draw_lr(lr_list, self.result_path)  # 绘图
                print('- lr:', self.optimizer.state_dict()
                      ['param_groups'][0]['lr'])

    time_elapsed = time.time() - since
    print_time(time_elapsed)


def train(self):
    begin = time.time()  # the start time

    original_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
    for epoch in range(self.num_epochs):
        print(f"\nepoch {epoch+1}/{self.num_epochs}")

        for phase in [PHASE_TRAIN, PHASE_EVAL]:
            if phase == PHASE_TRAIN:
                self.model.train()
            if phase == PHASE_EVAL:
                self.model.eval()

            top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
            top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
            losses = AverageMeter('Loss', ':.4e', Summary.NONE)
            progress = ProgressMeter(
                len(self.data_loaders[phase]),
                [losses, top1, top5],
                prefix=f'{phase}: ')

            # 这些数字都很小，相当于占位
            running_loss = 1e-5
            running_corrects = 1e-5
            class_correct = list(1e-5 for i in range(self.num_classes))
            class_total = list(1e-5 for i in range(self.num_classes))

            # model + data
            for idx, samples in enumerate(self.data_loaders[phase]):
                if idx % 10 == 0:
                    print(f"\r[{idx}/{len(self.data_loaders[phase])}]",
                          end="", flush=True)

                inputs, labels = samples
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, dim=1)
                    loss = self.criterion(outputs, labels)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                    losses.update(loss.item(), inputs.size(0))
                    top1.update(acc1[0], inputs.size(0))
                    top5.update(acc5[0], inputs.size(0))

                    # bp
                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()

                # 统计获得的数据
                running_loss += loss.item() * inputs.size(0)  # 本次epoch的损失
                running_corrects += torch.sum(preds == labels.data).item()
                correct_flag = (preds == labels.data)  # 那个位置的被预测对了
                for pos in range(len(labels)):
                    label = labels[pos]  # 获取该位置的标签
                    # 该标签预测正确的个数+1
                    class_correct[label] += correct_flag[pos].item()
                    class_total[label] += 1  # 该标签的总个数+1

                if idx % 100 == 0:
                    progress.display(idx)

            # # print each class acc and loss
            # print("\n")
            # for i in range(self.num_classes):
            #     class_acc = class_correct[i] / class_total[i] * 100
            #     print(f"acc of {i:2d} : {class_acc:.2f}%")
            epoch_loss = running_loss / self.dataset_sizes[phase]
            epoch_acc = running_corrects / self.dataset_sizes[phase]
            print(f"\n[{phase}] loss is {epoch_loss:.4f}, acc is {epoch_acc:.4f}")
            print(f"\n[{phase}] acc1 is {top1.avg:.4f}, acc5 is {top5.avg:.4f}")
            progress.display_summary()

            # save best epoch
            is_best = self.history.update(
                phase=phase, acc=epoch_acc, loss=epoch_loss)
            if is_best:
                print("\n[Feat] update best weights")
                state = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer,
                    "epoch": f"{epoch+1}",  # 第几个 epoch
                    "acc": f"{epoch_acc}"
                }
                cp_path = os.path.join(self.result_path, "checkpoint.pth")
                torch.save(state, cp_path)

            if phase == PHASE_EVAL:
                self.history.draw()
                adjust_learning_rate(self.optimizer, epoch, original_lr)
                print("\n[Info] lr is ", self.optimizer.state_dict()["param_groups"][0]["lr"])

    time_elapsed = time.time() - begin
    print_time(time_elapsed)


def print_time(time_elapsed):
    time_hour = time_elapsed // 3600
    time_minite = (time_elapsed % 3600) // 60
    time_second = time_elapsed % 60
    print(
        f"\nTraining complete in {time_hour:.0f}h {time_minite:.0f}m {time_second:.0f}s")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, original_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # Use 0.01 as the initial learning rate for AlexNet or VGG
    lr = original_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
