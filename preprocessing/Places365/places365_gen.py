import torch
import torchvision

dataset = torchvision.datasets.Places365(
    root="/mnt/hangzhou_116_homes/xwx/dataset/Places365",
    small=True,
    download=True,
)

it = iter(dataset)
print(next(it))
