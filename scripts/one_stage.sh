#!/bin/bash


python3 /mnt/nfs/xwx/model-doctor-xwx/trainers/pure_cls_train.py \
    --data_name cifar-10 \
    --lr 0.01 \
    --lr_scheduler cosine \
    --loss_type ce \
    --gpu_id 3 \
    > $(date "+%Y%m%d-%H%M%S").log
wait

python3 /mnt/nfs/xwx/model-doctor-xwx/trainers/pure_cls_train.py \
    --data_name cifar-10-lt-ir100 \
    --lr 0.01 \
    --lr_scheduler cosine \
    --loss_type ce \
    --gpu_id 3 \
    > $(date "+%Y%m%d-%H%M%S").log
wait


