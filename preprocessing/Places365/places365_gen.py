import torch
import torchvision

imagenet_data = torchvision.datasets.Places365(
    root="/nfs/xwx/dataset/Places365",
    small=True,
    download=True,
)