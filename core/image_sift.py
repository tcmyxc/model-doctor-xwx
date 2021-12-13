import sys
import os
import torch
import json

sys.path.append('/home/xwx/model-doctor-xwx') #205
import loaders
import models
from configs import config
from utils import file_util


class ImageSift:
    def __init__(self, class_nums, image_nums, is_high_confidence=True):

        # 每一行都是None，总共有 class_nums 行
        self.names = [[None for j in range(image_nums)] for i in range(class_nums)]
        self.scores = torch.zeros((class_nums, image_nums))
        self.nums = torch.zeros(class_nums, dtype=torch.long)
        self.is_high_confidence = is_high_confidence

    def __call__(self, outputs, labels, names):
        softmax = torch.nn.Softmax(dim=1)(outputs.detach())
        scores, predicts = torch.max(softmax, dim=1)

        if self.is_high_confidence:
            for i, label in enumerate(labels):
                if label == predicts[i]:
                    if self.nums[label] == self.scores.shape[1]:
                        score_min, index = torch.min(self.scores[label], dim=0)
                        if scores[i] > score_min:
                            self.scores[label][index] = scores[i]
                            self.names[label.item()][index.item()] = names[i]
                    else:
                        self.scores[label][self.nums[label]] = scores[i]
                        self.names[label.item()][self.nums[label].item()] = names[i]
                        self.nums[label] += 1
        else:
            for i, label in enumerate(labels):
                if self.nums[label] == self.scores.shape[1]:
                    score_max, index = torch.max(self.scores[label], dim=0)
                    if label == predicts[i]:  # TP-LS
                        if scores[i] < score_max:
                            self.scores[label][index] = scores[i]
                            self.names[label.item()][index.item()] = names[i]
                    else:  # TN-HS
                        if -scores[i] < score_max:
                            self.scores[label][index] = -scores[i]
                            self.names[label.item()][index.item()] = names[i]
                else:
                    if label == predicts[i]:  # TP-LS
                        self.scores[label][self.nums[label]] = scores[i]
                        self.names[label.item()][self.nums[label].item()] = names[i]
                        self.nums[label] += 1
                    else:  # TN-HS
                        self.scores[label][self.nums[label]] = -scores[i]
                        self.names[label.item()][self.nums[label].item()] = names[i]
                        self.nums[label] += 1

    def save_image(self, result_path):
        # print(self.scores)
        # print(self.nums)

        image_dir = os.path.join(config.data_cifar10, 'test')

        class_names = sorted([d.name for d in os.scandir(image_dir) if d.is_dir()])

        for label, image_list in enumerate(self.names):
            for image in image_list:
                class_name = class_names[label]

                src_path = os.path.join(image_dir, class_name, str(image))
                dst_path = os.path.join(result_path, 'images', class_name, str(image))
                file_util.copy_file(src_path, dst_path)


def sift_image(data_name, model_name, model_path, result_path):
    # device
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda:0')

    # config
    cfg = json.load(open('configs/config_trainer.json'))[data_name]

    # model
    model = models.load_model(model_name=model_name,
                              in_channels=cfg['model']['in_channels'],
                              num_classes=cfg['model']['num_classes'])
    model.load_state_dict(torch.load(model_path)['model'])
    model.to(device)
    model.eval()

    # data
    data_loader, _ = loaders.load_data(data_name=data_name,
                                       data_type='test')

    image_sift = ImageSift(class_nums=cfg['model']['num_classes'],
                           image_nums=20,
                           is_high_confidence=False)

    # forward
    for i, samples in enumerate(data_loader):
        print('\r[{}/{}]'.format(i, len(data_loader)), end='', flush=True)
        inputs, labels, names = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        image_sift(outputs=outputs, labels=labels, names=names)

    print('\n', end='', flush=True)
    image_sift.save_image(result_path)  # 保存低置信度图片


def main():
    data_name = 'cifar-10'
    model_name = 'resnet50'
    model_path = os.path.join(
        config.model_pretrained,
        model_name + '-20211208-101731', 
        'checkpoint.pth')
    result_path = os.path.join(
        config.output_result, 
        model_name + '-' + data_name, "low")
    print("\n==> result_path", result_path)
   
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    sift_image(data_name, model_name, model_path, result_path)


if __name__ == '__main__':
    main()
