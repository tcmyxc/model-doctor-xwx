# root = r"/mnt/hangzhou_116_homes/xwx"  # 204
root = r"/nfs/xwx"  # 205

# the result root dir
output_dir = root + '/model-doctor-xwx/output'

# image data dir
output_data = root + "/dataset"

# channel, mask
output_result = output_dir + '/result'

# image data
data_cifar10            = output_data + '/cifar10/images'
data_cifar100           = output_data + '/cifar100/images'
# data_cifar100_lt        = output_data + '/cifar-100-python'
data_mnist              = output_data + '/mnist/images'
data_fashion_mnist      = output_data + '/fashion_mnist/images'
data_svhn               = output_data + '/svhn/images'
data_stl10              = output_data + '/stl10/images'
data_mini_imagenet      = output_data + '/mini_imagenet/images'
data_mini_imagenet_temp = output_data + '/mini_imagenet/temp'
data_mini_imagenet_10   = output_data + '/mini_imagenet_10/images'

data_imagenet_lt        = output_data + "/ImageNet_LT"

data_inaturalist2018    = output_data + "/iNaturalist2018"

data_cifar10_lt_ir10    = output_data + "/cifar10_lt_ir10/images"
data_cifar10_lt_ir100   = output_data + "/cifar10_lt_ir100/images"

data_cifar100_lt_ir10   = output_data + "/cifar100_lt_ir10/images"
data_cifar100_lt_ir50   = output_data + "/cifar100_lt_ir50/images"
data_cifar100_lt_ir100  = output_data + "/cifar100_lt_ir100/images"
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

# result
result_masks_cifar10 = output_result + '/masks/cifar10'
result_masks_mnim10  = output_result + '/masks/mini_imagenet_10'
result_masks_mnim    = output_result + '/masks/mini_imagenet'
result_masks_stl10   = output_result + '/masks/stl10'
result_channels      = output_result + '/channels'

# model
output_model = output_dir + '/model'
model_pretrained = output_model + '/pretrained'
