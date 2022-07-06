#!/bin/bash

python3 /nfs/xwx/model-doctor-xwx/trainers/pure_cls_train.py \
    --data_name cifar-100-lt-ir10 \
    --lr 0.1 \
    --lr_scheduler custom \
    --loss_type ce \
    > $(date "+%Y%m%d-%H%M%S").log
wait

python3 /nfs/xwx/model-doctor-xwx/trainers/pure_cls_train.py \
    --data_name cifar-100-lt-ir50 \
    --lr 0.1 \
    --lr_scheduler custom \
    --loss_type ce \
    > $(date "+%Y%m%d-%H%M%S").log
wait

python3 /nfs/xwx/model-doctor-xwx/trainers/pure_cls_train.py \
    --data_name cifar-100-lt-ir100 \
    --lr 0.1 \
    --lr_scheduler custom \
    --loss_type ce \
    > $(date "+%Y%m%d-%H%M%S").log
wait

python3 /nfs/xwx/model-doctor-xwx/trainers/pure_cls_train.py \
    --data_name cifar-10-lt-ir10 \
    --lr 0.1 \
    --lr_scheduler custom \
    --loss_type ce \
    > $(date "+%Y%m%d-%H%M%S").log
wait

python3 /nfs/xwx/model-doctor-xwx/trainers/pure_cls_train.py \
    --data_name cifar-10-lt-ir100 \
    --lr 0.1 \
    --lr_scheduler custom \
    --loss_type ce \
    > $(date "+%Y%m%d-%H%M%S").log
wait

