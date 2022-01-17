import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import time
import os
import numpy as np

from core.grad_constraint import GradConstraint


class ClsTrainer:
    def __init__(self, model, criterion, optimizer, scheduler,
                 data_loaders, dataset_sizes, device, num_epochs,
                 model_path, checkpoint_path, num_classes=12):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_loaders = data_loaders
        self.dataset_sizes = dataset_sizes
        self.device = device
        self.num_epochs = num_epochs
        self.save_path = model_path
        self.num_classes = num_classes

        self.model.to(device)
        self.gc = GradConstraint(model=self.model,
                                 target_layer=self.model.features[40])
        if checkpoint_path is not None:
            print('-' * 40)
            print('Load checkpoint:', checkpoint_path)
            print('-' * 40)
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.history = TrainerHistory(best='acc',
                                      save_path=self.save_path)

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
                else:
                    self.model.eval()

                running_loss = 1e-5
                running_corrects = 1e-5
                class_correct = list(1e-5 for i in range(self.num_classes))
                class_total = list(1e-5 for i in range(self.num_classes))

                # ----------------------------------------
                # Iterate over data.
                # ----------------------------------------
                for i, samples in enumerate(self.data_loaders[phase]):
                    if i % 10 == 0:
                        print('\r[{}/{}]'.format(i, len(self.data_loaders[phase])), end='', flush=True)

                    inputs, labels = samples
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, dim=1)
                        loss = self.criterion(outputs, labels)



                        # backward
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # ----------------------------------------
                    # Calculate the loss and acc of each epoch
                    # ----------------------------------------
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()
                    correct_flag = (preds == labels.data)
                    for j in range(len(labels)):
                        label = labels[j]
                        class_correct[label] += correct_flag[j].item()
                        class_total[label] += 1

                for i in range(self.num_classes):
                    class_acc = 100 * class_correct[i] / class_total[i]
                    print('\rAccuracy of %2d : %2d %%' % (i, class_acc))
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects / self.dataset_sizes[phase]
                print('[{}] Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # ----------------------------------------
                # Save epoch data
                # ----------------------------------------
                is_best = self.history.update(phase=phase,
                                              acc=epoch_acc,
                                              loss=epoch_loss)
                if is_best:
                    print('- Update best weights')
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    state = {'model': self.model.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             'epoch': epoch,
                             'best': self.history.best}
                    path = os.path.join(self.save_path, 'checkpoint.pth')
                    torch.save(state, path)

                if phase == 'val':
                    self.history.draw()
                    self.scheduler.step()
                    print('- lr:', self.optimizer.state_dict()['param_groups'][0]['lr'])

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def check(self, model_path):
        phase = 'val'

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        print('-' * 40)
        print('Check data type:', phase)
        print('Load model from:', model_path)
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
        print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        for i in range(self.num_classes):
            print('Accuracy of %2d : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))
        time_elapsed = time.time() - since
        print('\rPhase:{} complete in {:.0f}m {:.0f}s'.format(phase, time_elapsed // 60, time_elapsed % 60))


class TrainerHistory(object):
    def __init__(self, best, save_path):
        assert best in ['loss', 'acc']
        self.best = best
        self.best_value = None

        self.save_path = save_path

        self.history = {'train_loss': [],
                        'train_acc': [],
                        'val_loss': [],
                        'val_acc': []}

    def update(self, phase, loss, acc):
        if phase == 'train':
            self.history['train_loss'].append(loss)
            self.history['train_acc'].append(acc)
        if phase == 'val':
            self.history['val_loss'].append(loss)
            self.history['val_acc'].append(acc)

            # best
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
