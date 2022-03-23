import matplotlib

# 在导入matplotlib库后，且在matplotlib.pyplot库被导入前加下面这句话，不然不起作用
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np
from enum import Enum

from loss.fl import focal_loss
from loss.efl import equalized_focal_loss
from loss.refl import reduce_equalized_focal_loss
from loss.rfl import reduced_focal_loss
from loss.dfl import dual_focal_loss

from trainers.cls_grad_trainer import draw_lr

CHECKPOINT_MODEL_NAME = "checkpoint.pth"
PHASE_TRAIN = "train"
PHASE_EVAL = "val"


class ClsTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, data_loaders,
                 dataset_sizes, device, num_epochs,
                 result_path, model_path, num_classes=10):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler  # 学习率调度
        self.data_loaders = data_loaders
        self.dataset_sizes = dataset_sizes
        self.device = device
        self.num_epochs = num_epochs
        self.num_classes = num_classes  # 类别数

        self.result_path = result_path
        if not os.path.exists(self.result_path):
            print("-" * 79, "\n[Info] the path does not exist, I will make it")
            os.makedirs(self.result_path)

        self.model.to(device)
        # 如果有预训练的模型，则直接加载预训练模型
        if model_path is not None:
            print("-" * 79, "\n[Info] load checkpoint:", model_path)
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint["model"])

        self.history = TrainerHistory(best="acc", save_path=self.result_path)

    def train(self):
        lr_list = []  # 记录学习率变化
        begin = time.time()  # the start time

        for epoch in range(self.num_epochs):
            epoch_begin_time = time.time()
            print(f"\nepoch {epoch+1}/{self.num_epochs}")  # log

            for phase in [PHASE_TRAIN, PHASE_EVAL]:
                if phase == PHASE_TRAIN:
                    self.model.train()
                if phase == PHASE_EVAL:
                    self.model.eval()

                top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
                top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

                # 这些数字都很小，相当于占位
                running_loss = 1e-5
                running_corrects = 1e-5
                class_correct = list(1e-5 for i in range(self.num_classes))
                class_total = list(1e-5 for i in range(self.num_classes))

                # model + data
                for idx, samples in enumerate(self.data_loaders[phase]):
                    if idx % 10 == 0:
                        print(f"\r[{idx}/{len(self.data_loaders[phase])}]", end="", flush=True)

                    inputs, labels, _ = samples
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(True):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, dim=1)
                        loss = self.criterion(outputs, labels)  # ce loss
                        # loss = reduce_equalized_focal_loss(outputs, labels, threshold=0.5)  # refl
                        # loss = focal_loss(outputs, labels) # fl
                        # loss = equalized_focal_loss(outputs, labels)  # efl
                        # loss = reduced_focal_loss(outputs, labels)  # rfl
                        # loss = dual_focal_loss(outputs, labels)  # dfl

                        # measure accuracy and record loss
                        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
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
                        class_correct[label] += correct_flag[pos].item()  # 该标签预测正确的个数+1
                        class_total[label] += 1  # 该标签的总个数+1

                # print each class acc and loss
                print("\n")
                for i in range(self.num_classes):
                    class_acc = class_correct[i] / class_total[i] * 100
                    print(f"acc of {i:2d} : {class_acc:.2f}%")
                
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects / self.dataset_sizes[phase]
                print(f"\n[{phase}] loss is {epoch_loss:.4f}, acc is {epoch_acc:.4f}")
                print(f"\n[{phase}] acc1 is {top1.avg:.2f}%, acc5 is {top5.avg:.2f}%")
                print(f"[{phase}] err1 is {(100-top1.avg):.2f}%, err5 is {(100-top5.avg):.2f}%")

                # save best epoch
                is_best = self.history.update(phase=phase, acc=epoch_acc, loss=epoch_loss)
                if is_best:
                    print("\n[Feat] update best weights")
                    state = {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer,
                        "epoch": epoch,  # 第几个 epoch
                        "acc": f"{epoch_acc}"
                    }
                    cp_path = os.path.join(self.result_path, "checkpoint.pth")
                    torch.save(state, cp_path)
                
                if phase == PHASE_EVAL:
                    self.history.draw()

                    cur_lr = float(self.optimizer.state_dict()['param_groups'][0]['lr'])
                    lr_list.append(cur_lr)
                    draw_lr(lr_list, self.result_path)  # 绘图
                    print("\n[Info] lr is ", self.optimizer.state_dict()["param_groups"][0]["lr"])
                    
                    self.scheduler.step()
            
            # 打印一个完整的训练加测试花费多少时间
            print_time(time.time()-epoch_begin_time, epoch=True)

        # 打印训练总共花费时间
        print_time(time.time() - begin)

    def check(self):
        phase = PHASE_EVAL

        cp_path = os.path.join(self.result_path, CHECKPOINT_MODEL_NAME)
        if not os.path.exists(cp_path):
            print("=" * 42)
            print("模型文件的路径不存在, 请检查")
            return
        state = torch.load(cp_path)
        # print("bets res is", state["epoch"], "epoch, optimizer is", state["optimizer"])
        # return
        self.model.load_state_dict(state['model'])
        self.model.eval()

        print('-' * 40)
        print('Check data type:', phase)
        print('Load model from:', self.result_path)
        print('Data size:', self.dataset_sizes[phase])
        print('-' * 40)

        since = time.time()

        running_loss = 1e-5
        running_corrects = 1e-5
        class_correct = list(1e-5 for i in range(self.num_classes))
        class_total = list(1e-5 for i in range(self.num_classes))

        # Iterate over data.
        for i, samples in enumerate(self.data_loaders[phase]):
            if i % 10 == 0:
                print('\r[{}/{}]'.format(i, len(self.data_loaders[phase])), end='', flush=True)

            inputs, labels = samples
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward
            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

            correct_flag = (preds == labels.data)
            for j in range(len(inputs)):
                label = labels[j]
                class_correct[label] += correct_flag[j].item()
                class_total[label] += 1

        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc = running_corrects / self.dataset_sizes[phase]
        print('\rTest Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        for i in range(self.num_classes):
            print('Accuracy of {:2d} : {:.2f}%'.format(i, 100 * class_correct[i] / class_total[i]))
        
        print_time(time.time() - since)

class TrainerHistory(object):
    def __init__(self, best, save_path):
        assert best in ['loss', 'acc']
        self.best = best
        self.best_value = None

        self.save_path = save_path

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def update(self, phase, loss, acc):
        if phase == 'train':
            self.history['train_loss'].append(loss)
            self.history['train_acc'].append(acc)
        if phase == 'val':
            self.history['val_loss'].append(loss)
            self.history['val_acc'].append(acc)

            # best，保存最低的loss和最优的准确度
            if self.best == 'loss' and (self.best_value is None or loss <= self.best_value):
                self.best_value = loss
                return True
            if self.best == 'acc' and (self.best_value is None or acc >= self.best_value):
                self.best_value = acc
                return True

        return False

    def draw(self):
        # save history
        np.save(os.path.join(self.save_path, 'model.npy'), self.history)

        # draw history
        num_epochs = len(self.history['train_loss'])

        plt.plot(range(1, num_epochs + 1), self.history['train_loss'], 'r', label='train loss')
        plt.plot(range(1, num_epochs + 1), self.history['train_acc'], 'g', label='train acc')
        plt.plot(range(1, num_epochs + 1), self.history['val_loss'], 'b', label='val loss')
        plt.plot(range(1, num_epochs + 1), self.history['val_acc'], 'k', label='val acc')

        plt.title("Acc and Loss of each epoch")
        plt.xlabel("Training Epochs")
        plt.ylabel("Acc Loss")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'model.jpg'))
        plt.clf()


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


def print_time(time_elapsed, epoch=False):
    time_hour = time_elapsed // 3600
    time_minite = (time_elapsed % 3600) // 60
    time_second = time_elapsed % 60
    if epoch:
        print(f"\nCurrent epoch take time: {time_hour:.0f}h {time_minite:.0f}m {time_second:.0f}s")
    else:
        print(f"\nAll complete in {time_hour:.0f}h {time_minite:.0f}m {time_second:.0f}s")