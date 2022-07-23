#!/bin/bash


python3 /nfs/xwx/model-doctor-xwx/trainers/pure_cls_train.py \
    --data_name cifar-10 \
    --lr 0.01 \
    --lr_scheduler cosine \
    --loss_type ce \
    --gpu_id 1 \
    > $(date "+%Y%m%d-%H%M%S").log
wait

python3 /nfs/xwx/model-doctor-xwx/trainers/pure_cls_train.py \
    --data_name cifar-10-lt-ir100 \
    --lr 0.01 \
    --lr_scheduler cosine \
    --loss_type ce \
    --gpu_id 1 \
    > $(date "+%Y%m%d-%H%M%S").log
wait

python3 /nfs/xwx/model-doctor-xwx/trainers/pure_cls_train.py \
    --data_name cifar-10 \
    --lr 0.01 \
    --lr_scheduler cosine \
    --loss_type ce \
    --gpu_id 1 \
    > $(date "+%Y%m%d-%H%M%S").log
wait
