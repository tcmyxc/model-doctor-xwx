import sys
import os
import torch
import json

# 代码路径，让python解释器可以找到路径
sys.path.append('/home/xwx/model-doctor-xwx/')
import loaders
import models
from configs import config
from utils import file_util

import numpy as np
import cv2

import torch
from torchvision.transforms import Compose, Normalize, ToTensor


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        # 获取forward时target_layer的输出
        target_layer.register_forward_hook(self.save_activation)
        # 获取backward时target_layer的输出梯度
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)
        # print('acti_output', output.shape)

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        self.gradients = [grad_output[0]] + self.gradients
        # print('grad_output', grad_output[0].shape)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


class BaseCAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        return output[:, target_category]

    def __call__(self, input_tensor, target_category=None, is_whole=True):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        # model计算结果（score*12）
        output = self.activations_and_grads(input_tensor)
        # print('output', output)

        # 找出score最大的index作为target category
        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())
            print('===', target_category)

        # 把模型参数的梯度设为0
        self.model.zero_grad()
        # 利用index取出score作为loss
        loss = self.get_loss(output, target_category)
        # print('loss', loss)
        # 利用score进行反向传播
        loss.backward(retain_graph=True)

        # 获取forward激活
        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        # print('activations', activations)
        # 获取backward梯度
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        # print('grads', grads)

        # 用target layer的梯度mean作为每一层feature map的weight（shape即为fm层数）
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        # print('weights', weights.shape)

        if is_whole:
            return self.cam_whole(activations, weights)
        else:
            return self.cam_part(activations, weights)

    def cam_whole(self, activations, weights):
        # 定义一个和feature map大小一致的cam（size*size）
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        # 用每一层的weight去乘每一层fm的激活（输出）再叠加在一起
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # 将cam中的负值都变为0（类似于ReLU)
        cam = np.maximum(cam, 0)
        # # 将cam resize为输入图的大小
        # cam = cv2.resize(cam, input_tensor.shape[2:][::-1])
        # 对cam的值进行标准化（0-1）
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

    def cam_part(self, activations, weights):
        # # 定义一个和feature map一致的cam（c*size*size）
        # cams = np.zeros(activations.shape, dtype=np.float32)
        #
        # for i, w in enumerate(weights):
        #     # 用每一层的weight去乘每一层fm的激活（输出）
        #     cam = w * activations[i, :, :]
        #
        #     # 将cam中的负值都变为0（类似于ReLU)
        #     cam = np.maximum(cam, 0)
        #     # # 将cam resize为输入图的大小
        #     # cam = cv2.resize(cam, input_tensor.shape[2:][::-1])
        #     # 对cam的值进行标准化（0-1）
        #     cam = cam - np.min(cam)
        #     cam = cam / (np.max(cam) + 1e-10)
        #
        #     print(np.max(cam))
        #     cams[i] = cam
        # return cams

        cams = np.zeros(activations.shape, dtype=np.float32)
        for i, w in enumerate(weights):
            # 用每一层的weight去乘每一层fm的激活（输出）
            cam = w * activations[i, :, :]

            # 将cam中的负值都变为0（类似于ReLU)
            cam = np.maximum(cam, 0)

            cams[i] = cam

        cams_min = np.min(cams)
        cams_max = np.max(cams)

        for i, cam in enumerate(cams):
            # 对每一层cam的值进行标准化（0-1）
            cam = cam - np.min(cams_min)
            cam = cam / np.max(cams_max)

            cams[i] = cam
        return cams


class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(GradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor,
                        target_category,
                        activations, grads):
        return np.mean(grads, axis=(1, 2))


def show_cam_on_image(img: np.ndarray, mask: np.ndarray, is_whole=True, with_img=True) -> np.ndarray:
    if is_whole:
        mask = cv2.resize(mask, img.shape[0:2])
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        if with_img:
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)
            return np.uint8(255 * cam)
        else:
            print(heatmap.shape)
            return heatmap
    else:
        results = np.zeros((mask.shape[0], img.shape[0], img.shape[1], img.shape[2]))
        for i, m in enumerate(mask):
            m = cv2.resize(m, img.shape[0:2])
            heatmap = cv2.applyColorMap(np.uint8(255 * m), cv2.COLORMAP_JET)
            if with_img:
                heatmap = np.float32(heatmap) / 255
                cam = heatmap + np.float32(img)
                cam = cam / np.max(cam)
                results[i] = np.uint8(255 * cam)
            else:
                results[i] = heatmap
        return results


def image_process(image_path):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (32, 32))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465),
                  (0.2023, 0.1994, 0.2010))
    ])(rgb_img).unsqueeze(0)
    return input_tensor, rgb_img


def label_process(class_name):
    image_dir = os.path.join(config.data_cifar10, 'test')

    class_names = sorted([d.name for d in os.scandir(image_dir) if d.is_dir()])
    class_dic = {class_names[i]: i for i in range(len(class_names))}

    label = class_dic[class_name]
    return label


def save_ori_image(save_path, save_name, ori_path):
    rgb_img = cv2.imread(ori_path, 1)
    rgb_img = cv2.resize(rgb_img, (32, 32))
    filename = '{}/{}.png'.format(save_path, save_name)
    cv2.imwrite(filename, rgb_img)


def save_cam_image(save_path, save_name, img, is_whole=True):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if is_whole:
        filename = '{}/{}.png'.format(save_path, save_name)
        cv2.imwrite(filename, img)
    else:
        for i, m in enumerate(img):
            filename = '{}/{}_{}.png'.format(save_path, save_name, i)
            cv2.imwrite(filename, m)


def grad_cam(model, module, image_path, class_name):
    is_whole = True

    cam = GradCAM(model=model,
                  target_layer=module,
                  use_cuda=True)

    input_tensor, rgb_img = image_process(image_path)
    target_category = label_process(class_name)

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        is_whole=is_whole)

    cam_image = show_cam_on_image(img=rgb_img,
                                  mask=grayscale_cam,
                                  is_whole=is_whole,
                                  with_img=is_whole)
    return cam_image


def main():
    data_name = 'cifar-10'
    model_name = 'resnet50'
    model_layers = [-1]
    model_path = os.path.join(config.output_model, 'gc', model_name + '-20211208-101731-pos-activation', 'checkpoint.pth')
    
    images_path = os.path.join(
        config.output_result, 
        model_name + '-' + data_name, 
        "high", 'images')
    
    result_path = os.path.join(
        config.output_result, 
        model_name + '-' + data_name + '-pos-activation',
        "high", 'cams')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    cfg = json.load(open('configs/config_trainer.json'))[data_name]

    model = models.load_model(model_name=model_name,
                              in_channels=cfg['model']['in_channels'],
                              num_classes=cfg['model']['num_classes'])
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()

    module = models.load_modules(model=model,
                                 model_name=model_name,
                                 model_layers=model_layers)[0]

    images_list = read_images_list(images_path)
    for i, class_name in enumerate(images_list):
        print('-' * 40)
        for i, image_name in enumerate(images_list[class_name]):
            image_path = os.path.join(images_path, class_name, image_name)
            save_path = os.path.join(result_path, class_name)
            cam = grad_cam(model, module, image_path, class_name)
            # TODO rename
            save_cam_image(save_path=save_path, save_name='{}_channel'.format(i), img=cam, is_whole=True)
            # save_cam_image(save_path=save_path, save_name='{}_gradcam'.format(i), img=cam, is_whole=True)
            # save_ori_image(save_path=save_path, save_name='{}_original'.format(i), ori_path=image_path)
            print(cam.shape)


def read_images_list(path):
    images = {}
    for root, _, files in os.walk(path):
        if len(files) != 0:
            class_name = root.split('/')[-1]
            images[class_name] = files
    print(images)
    return images


if __name__ == '__main__':
    main()
