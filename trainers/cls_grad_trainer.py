import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import time
import os
import numpy as np

from core.grad_constraint import GradConstraint
from core.noise_augment import GradIntegral


class ClsGradTrainer:
    def __init__(self, model, modules, device, criterion, optimizer, scheduler,
                 data_loaders, dataset_sizes, num_epochs, num_classes,
                 result_path, model_path, channel_paths):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_loaders = data_loaders
        self.dataset_sizes = dataset_sizes
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.result_path = result_path

        self.model.to(device)

        self.gi = GradIntegral(model=model, modules=modules)
        self.gc = GradConstraint(model=model, modules=modules, channel_paths=channel_paths)

        if model_path is not None:
            print('-' * 40)
            print('LOAD CHECKPOINT:', model_path)
            print('-' * 40)
            self.model.load_state_dict(torch.load(model_path)['model'])

        self.history = TrainerHistory(best='acc', save_path=self.result_path)

    def train(self):
        since = time.time()

        # ----------------------------------------
        # Each epochs.
        # ----------------------------------------
        for epoch in range(self.num_epochs):
            print('\nEpoch {}/{}'.format(epoch, self.num_epochs - 1))

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    self.gi.add_noise()
                else:
                    self.model.eval()
                    self.gi.remove_noise()

                running_loss_cls = 1e-5
                running_loss_gc = 1e-5
                running_corrects = 1e-5
                class_correct = list(1e-5 for i in range(self.num_classes))
                class_total = list(1e-5 for i in range(self.num_classes))

                # ----------------------------------------
                # Iterate over data.
                # ----------------------------------------
                for i, samples in enumerate(self.data_loaders[phase]):
                    inputs, labels, masks = samples
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

                        loss_channel = self.gc.loss_channel(outputs=outputs, labels=labels)
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
                print("\n")
                for i in range(self.num_classes):
                    class_acc = 100 * class_correct[i] / class_total[i]
                    print('\rAccuracy of %2d : %2d %%' % (i, class_acc))
                epoch_loss_cls = running_loss_cls / self.dataset_sizes[phase]
                epoch_loss_gc = running_loss_gc / self.dataset_sizes[phase]
                epoch_acc = running_corrects / self.dataset_sizes[phase]
                print('[{}] Loss CLS: {:.4f} Loss GC: {:.4f} Acc: {:.4f}'
                      .format(phase, epoch_loss_cls, epoch_loss_gc, epoch_acc))

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
                        'history': self.history.history
                    }
                    path = os.path.join(self.result_path, 'checkpoint.pth')
                    torch.save(state, path)

                if phase == 'val':
                    self.history.draw()
                    self.scheduler.step()
                    print('- lr:', self.optimizer.state_dict()['param_groups'][0]['lr'])

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def check(self):
        phase = 'val'

        state = torch.load(self.result_path + '/checkpoint.pth')
        self.model.load_state_dict(state['model'])
        self.model.eval()

        print('-' * 40)
        print('Check data type:', phase)
        print('Load model from:', self.result_path)
        print('Data size:', self.dataset_sizes[phase])
        print('-' * 40)

        since = time.time()

        running_loss = 0.0
        running_corrects = 0.0
        class_correct = list(1e-10 for i in range(self.num_classes))
        class_total = list(1e-10 for i in range(self.num_classes))

        # Iterate over data.
        for i, samples in enumerate(self.data_loaders[phase]):
            if i % 10 == 0:
                print('\r{}/{}'.format(i, len(self.data_loaders[phase])), end='', flush=True)

            inputs, labels, _ = samples
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # masks = masks.to(self.device)

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
        print('\rTest CLS Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        for i in range(self.num_classes):
            print('Accuracy of {:2d} : {:.2f}%'.format(i, 100 * class_correct[i] / class_total[i]))
        time_elapsed = time.time() - since
        print('\rPhase:{} complete in {:.0f}m {:.0f}s'.format(phase, time_elapsed // 60, time_elapsed % 60))


class TrainerHistory(object):
    def __init__(self, best, save_path):
        assert best in ['loss', 'acc']
        self.best = best
        self.best_value = None

        self.save_path = save_path

        self.history = {'train_loss_cls': [],
                        'train_loss_gc': [],
                        'train_acc': [],
                        'val_loss_cls': [],
                        'val_loss_gc': [],
                        'val_acc': []}

    def update(self, phase, loss_cls, loss_gc, acc):
        if phase == 'train':
            self.history['train_loss_cls'].append(loss_cls)
            self.history['train_loss_gc'].append(loss_gc)
            self.history['train_acc'].append(acc)
        if phase == 'val':
            self.history['val_loss_cls'].append(loss_cls)
            self.history['val_loss_gc'].append(loss_gc)
            self.history['val_acc'].append(acc)

            # best
            if self.best == 'loss' and (self.best_value is None or loss_cls <= self.best_value):
                self.best_value = loss_cls
                return True
            if self.best == 'acc' and (self.best_value is None or acc >= self.best_value):
                self.best_value = acc
                return True

        return False

    def draw(self):
        # save history
        np.save(os.path.join(self.save_path, 'model.npy'), self.history)

        # draw history
        num_epochs = len(self.history['train_loss_cls'])

        plt.plot(range(1, num_epochs + 1), self.history['train_loss_cls'], 'r', label='train loss_cls')
        plt.plot(range(1, num_epochs + 1), self.history['train_loss_gc'], 'deeppink', label='train loss_gc')
        plt.plot(range(1, num_epochs + 1), self.history['train_acc'], 'g', label='train acc')
        plt.plot(range(1, num_epochs + 1), self.history['val_loss_cls'], 'b', label='val loss_cls')
        plt.plot(range(1, num_epochs + 1), self.history['val_loss_gc'], 'blueviolet', label='val loss_gc')
        plt.plot(range(1, num_epochs + 1), self.history['val_acc'], 'k', label='val acc')

        plt.title("Acc and Loss of each epoch")
        plt.xlabel("Training Epochs")
        plt.ylabel("Acc Loss")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'model.jpg'))
        plt.clf()
