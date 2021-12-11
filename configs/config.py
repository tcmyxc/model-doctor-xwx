root = r'/nfs3-p1/hjc/classification'


# ----------------------------------------
# original datasets
# ----------------------------------------
datasets_dir = root + '/datasets'

datasets_coco = datasets_dir + '/COCO'
datasets_CIFAR_10 = datasets_dir + '/cifar-10-batches-py'
datasets_CIFAR_100 = '/workspace/classification/output/data/cifar100/images/cifar-100-python'
datasets_STL_10 = datasets_dir + '/stl10_binary'
datasets_MNIST = '/workspace/classification/output/data/mnist/images/MNIST/raw'
datasets_FASHION_MNIST = '/workspace/classification/output/data/fashion_mnist/images/FashionMNIST/raw'

# image data
output_data = "/home/xwx/model-doctor-xwx/data"
data_cifar10 = output_data + '/cifar10/images'
data_cifar100 = output_data + '/cifar100/images'
data_mnist = output_data + '/mnist/images'
data_fashion_mnist = output_data + '/fashion_mnist/images'
data_svhn = output_data + '/svhn/images'
data_stl10 = output_data + '/stl10/images'
data_mini_imagenet = output_data + '/mini_imagenet/images'
data_mini_imagenet_temp = output_data + '/mini_imagenet/temp'
data_mini_imagenet_10 = output_data + '/mini_imagenet_10/images'
# # data coco
# coco_images = output_data + '/coco/images'
# coco_images_1 = output_data + '/coco_6x2/images1'
# coco_images_2 = output_data + '/coco_6x2/images2'
# coco_masks = output_data + '/coco/masks'
# coco_masks_processed_15 = output_data + '/coco/masks_processed_15'
# coco_masks_processed_32 = output_data + '/coco/masks_processed_32'

# ----------------------------------------
# output，放在自己的目录下面
# ----------------------------------------
output_dir = '/home/xwx/model-doctor-xwx/output'

# result
output_result = output_dir + '/result'
result_masks_cifar10 = output_result + '/masks/cifar10'
result_masks_mnim10 = output_result + '/masks/mini_imagenet_10'
result_masks_mnim = output_result + '/masks/mini_imagenet'
result_masks_stl10 = output_result + '/masks/stl10'
result_channels = output_result + '/channels'

# model
output_model = output_dir + '/model'
model_pretrained = output_model + '/pretrained'
