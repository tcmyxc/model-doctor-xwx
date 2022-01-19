import torch
import torchvision

imagenet_data = torchvision.datasets.Places365(
    root="/nfs/xwx/model-doctor-xwx/data/Places365",
    small=True,
    download=True,
)