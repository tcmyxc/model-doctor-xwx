#!/bin/bash


python3 /nfs/xwx/model-doctor-xwx/trainers/im_use_balance_data.py \
    --data_name cifar-10 \
    --epochs 200 \
    --model_path /nfs/xwx/model-doctor-xwx/output/model/pretrained/resnet32/cifar-10-lt-ir100/lr0.01/cosine_lr_scheduler/ce_loss/2022-07-15_17-27-58/best-model-acc0.7144.pth \
    > $(date "+%Y%m%d-%H%M%S").log
wait
