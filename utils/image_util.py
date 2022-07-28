import matplotlib

matplotlib.use('AGG')
import os
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import cv2

import seaborn as sns


def show_torch_images(images, mode=None):
    img = torchvision.utils.make_grid(images, pad_value=1)
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)), cmap=mode)
    plt.show()


def show_cv(image, end=0, name=None):
    cv2.imshow('test', image)
    if end:
        cv2.waitKey(0)


def save_cv(img, path):
    print(path)
    img_dir, _ = os.path.split(path)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    cv2.imwrite(path, img)


def cv_show(image, end=0, name=None):
    cv2.imshow('test', image)
    if end:
        cv2.waitKey(0)


def cv_save(img, path):
    print(path)
    img_dir, _ = os.path.split(path)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    cv2.imwrite(path, img)


# def deprocess_image(img):
#     """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
#     img = img - np.mean(img)
#     img = img / (np.std(img) + 1e-5)
#     img = img * 0.1
#     img = img + 0.5
#     img = np.clip(img, 0, 1)
#     return np.uint8(img * 255)

def deprocess_image(img, std, mean):
    import torch
    t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 32, 32).numpy()
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 32, 32).numpy()
    img = img * t_std + t_mean
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def view_grads(grads, fig_w, fig_h, fig_path):
    f, ax = plt.subplots(figsize=(fig_w, fig_h), ncols=1)
    ax.set_xlabel('convolutional kernel')
    ax.set_ylabel('category')
    sns.heatmap(grads, annot=False, ax=ax)
    plt.savefig(fig_path, bbox_inches='tight')
    # plt.show()
    plt.clf()
    plt.close()

    # plt.xlabel('convolutional kernel')
    # plt.ylabel('category')
    # sns.heatmap(grads)
    # plt.savefig(fig_path, bbox_inches='tight')
    # plt.clf()


def heatmap(vals, fig_path, fig_w=None, fig_h=None, annot=False):
    if fig_w is None:
        fig_w = vals.shape[1]
    if fig_h is None:
        fig_h = vals.shape[0]

    f, ax = plt.subplots(figsize=(fig_w, fig_h), ncols=1)
    sns.heatmap(vals, ax=ax, annot=annot)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def scatter(vals, fig_path):
    x = [i for i in range(len(vals[0]))]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    vals = [vals[0]]
    for val in vals:
        p1 = ax.scatter(x, val, marker='.', color='black', s=8)
    # plt.show()
    plt.savefig(fig_path)
    plt.clf()
