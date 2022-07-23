#!/bin/bash


# python3 /nfs/xwx/model-doctor-xwx/trainers/retrain.py \
#     --data_name cifar-10-lt-ir100 \
#     --lr 0.1 \
#     --lr_scheduler custom \
#     --loss_type bsl \
#     --model_path /nfs/xwx/model-doctor-xwx/output/model/pretrained/resnet32/cifar-10-lt-ir100/lr0.1/custom_lr_scheduler/bsl_loss/2022-07-14_21-13-14/best-model-acc0.8245.pth \
#     --gpu_id 1 \
#     > $(date "+%Y%m%d-%H%M%S").log
# wait

# python3 /nfs/xwx/model-doctor-xwx/trainers/retrain.py \
#     --data_name cifar-10-lt-ir100 \
#     --lr 0.01 \
#     --lr_scheduler custom \
#     --loss_type bsl \
#     --model_path /nfs/xwx/model-doctor-xwx/output/model/pretrained/resnet32/cifar-10-lt-ir100/lr0.1/custom_lr_scheduler/bsl_loss/2022-07-14_21-13-14/best-model-acc0.8245.pth \
#     --gpu_id 1 \
#     > $(date "+%Y%m%d-%H%M%S").log
# wait


python3 /nfs/xwx/model-doctor-xwx/trainers/retrain.py \
    --data_name cifar-10-lt-ir100 \
    --lr 0.001 \
    --lr_scheduler custom \
    --loss_type bsl \
    --model_path /nfs/xwx/model-doctor-xwx/output/model/pretrained/resnet32/cifar-10-lt-ir100/lr0.1/custom_lr_scheduler/bsl_loss/2022-07-14_21-13-14/best-model-acc0.8245.pth \
    --gpu_id 2 \
    > $(date "+%Y%m%d-%H%M%S").log
wait