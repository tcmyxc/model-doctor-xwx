from ensurepip import version
import torch
import torchvision

imagenet_data = torchvision.datasets.INaturalist(
    root="/nfs/xwx/model-doctor-xwx/data/iNaturalist",
    version="2018",
    download=True,
)