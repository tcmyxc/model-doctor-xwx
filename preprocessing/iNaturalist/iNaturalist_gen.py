import torchvision

dataset = torchvision.datasets.INaturalist(
    root="/nfs/xwx/dataset/iNaturalist2018",
    version="2018",
    download=True,
)

img_iter = iter(dataset)
print(next(img_iter))