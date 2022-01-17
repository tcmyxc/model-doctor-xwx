import sys

sys.path.append('/workspace/classification/code/')  # zjl

import os
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
import seaborn as sns
import matplotlib.pyplot as plt
import json

import models
from configs import config
from core.grad_constraint import HookModule


def save_heatmap(save_path, save_name, heatmap, is_whole=True):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if is_whole:
        filename = '{}/{}.png'.format(save_path, save_name)
        cv2.imwrite(filename, heatmap)
    else:
        for i, m in enumerate(heatmap):
            filename = '{}/{}_{}.png'.format(save_path, save_name, i)
            cv2.imwrite(filename, m)


def gen_heatmap(grad, img=None, is_whole=True):
    grad = torch.abs(grad)  # grad processing
    grad = grad.detach().numpy()[0, :]
    if is_whole:
        grad = np.sum(grad, axis=0)  # whole -> sum
        grad = grad - np.min(grad)
        grad = grad / np.max(grad)
        grad = cv2.resize(grad, (224, 224))

        heatmap = cv2.applyColorMap(np.uint8(255 * grad), cv2.COLORMAP_JET)  # gen heatmap
        if img is not None:
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap + np.float32(img)
            heatmap = heatmap / np.max(heatmap)
            heatmap = np.uint8(255 * heatmap)
        else:
            return heatmap
        return heatmap
    else:
        heatmaps = np.zeros((grad.shape[0], 224, 224, 3), dtype=np.float32)  # part -> inter
        grad_min = np.min(grad)
        grad_max = np.max(grad)
        for i, g in enumerate(grad):
            g = g - grad_min
            g = g / grad_max
            g = cv2.resize(g, (224, 224))

            heatmap = cv2.applyColorMap(np.uint8(255 * g), cv2.COLORMAP_JET)
            if img is not None:
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap + np.float32(img)
                heatmap = heatmap / np.max(heatmap)
                heatmaps[i] = np.uint8(255 * heatmap)
            else:
                heatmaps[i] = heatmap
        return heatmaps


def image_process(image_path):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])(rgb_img).unsqueeze(0)
    return input_tensor, rgb_img


def label_process(class_name):
    # class_dic = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    #              'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9} # cifar
    class_dic = {'n01770081': 0, 'n02091831': 1, 'n02108089': 2, 'n02687172': 3, 'n04251144': 4,
                 'n04389033': 5, 'n04435653': 6, 'n04443257': 7, 'n04515003': 8, 'n07747607': 9}
    label = torch.tensor(class_dic[class_name]).unsqueeze(0)
    return label


def calculate_grad(model, module, labels, inputs):
    module = HookModule(model=model, module=module)

    outputs = model(inputs)
    softmax = torch.nn.Softmax(dim=1)(outputs)
    scores, predicts = torch.max(softmax, 1)
    print('=== forward ===>', predicts, scores)

    nll_loss = torch.nn.NLLLoss()(outputs, labels)
    grads = module.grads(outputs=-nll_loss, inputs=module.activations,
                         retain_graph=True, create_graph=False)
    nll_loss.backward()  # to release graph

    return grads


def main():
    data_name = 'mini-imagenet-10'
    model_name = 'vgg16'
    model_layers = [-1]
    model_path = os.path.join(config.model_pretrained, model_name + '_08241356', 'checkpoint.pth')
    result_path = os.path.join(config.output_result, model_name + '_08241356')
    cfg = json.load(open('configs/config_trainer.json'))[data_name]

    model = models.load_model(model_name=model_name,
                              in_channels=cfg['model']['in_channels'],
                              num_classes=cfg['model']['num_classes'])
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()

    module = models.load_modules(model=model,
                                 model_name=model_name,
                                 model_layers=model_layers)[0]

    # test_images = {
    #     'n01770081': 'n0177008100000044.jpg',
    #     'n02091831': 'n0209183100000085.jpg',
    #     'n02108089': 'n0210808900000033.jpg',
    #     'n02687172': 'n0268717200000082.jpg',
    #     'n04251144': 'n0425114400000011.jpg',
    #     'n04389033': 'n0438903300000003.jpg',
    #     'n04435653': 'n0443565300000236.jpg',
    #     'n04443257': 'n0444325700000062.jpg',
    #     'n04515003': 'n0451500300000023.jpg',
    #     'n07747607': 'n0774760700000167.jpg',
    # }

    test_images = {
        'n01770081': 'n0177008100000075.jpg',
        'n02091831': 'n0209183100000171.jpg',
        'n02108089': 'n0210808900000206.jpg',
        'n02687172': 'n0268717200000050.jpg',
        'n04251144': 'n0425114400000079.jpg',
        'n04389033': 'n0438903300000100.jpg',
        'n04435653': 'n0443565300000079.jpg',
        'n04443257': 'n0444325700000038.jpg',
        'n04515003': 'n0451500300000148.jpg',
        'n07747607': 'n0774760700000034.jpg',
    }

    for i, name in enumerate(test_images):
        print('-' * 40)
        class_name = name
        image_name = test_images[name]
        save_path = os.path.join(result_path, 'grad response', 'high confidence')
        image_path = '{}/images_hc/{}/{}'.format(result_path, class_name, image_name)

        labels = label_process(class_name)
        inputs, images = image_process(image_path)

        grads = calculate_grad(model, module, labels, inputs)

        heatmap = gen_heatmap(grad=grads, img=images, is_whole=True)
        save_heatmap(save_path=save_path, save_name=i, heatmap=heatmap, is_whole=True)


if __name__ == '__main__':
    main()
