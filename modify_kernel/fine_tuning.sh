#!/bin/bash

python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --lr_scheduler custom \
    --loss_type refl \
    > $(date "+%Y%m%d-%H%M%S").log &
wait

python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --loss_type ce \
    --lr_scheduler custom \
    > $(date "+%Y%m%d-%H%M%S").log &
wait


python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --loss_type fl \
    --lr_scheduler custom \
    > $(date "+%Y%m%d-%H%M%S").log &
wait

# lr 0.1
python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --lr_scheduler custom \
    --lr 0.1 \
    --loss_type refl \
    > $(date "+%Y%m%d-%H%M%S").log &
wait

python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --loss_type ce \
    --lr_scheduler custom \
    --lr 0.1 \
    > $(date "+%Y%m%d-%H%M%S").log &
wait


python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --loss_type fl \
    --lr_scheduler custom \
    --lr 0.1 \
    > $(date "+%Y%m%d-%H%M%S").log &
wait

# lr 0.01
python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --lr_scheduler custom \
    --lr 0.01 \
    --loss_type refl \
    > $(date "+%Y%m%d-%H%M%S").log &
wait

python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --loss_type ce \
    --lr_scheduler custom \
    --lr 0.01 \
    > $(date "+%Y%m%d-%H%M%S").log &
wait


python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --loss_type fl \
    --lr_scheduler custom \
    --lr 0.01 \
    > $(date "+%Y%m%d-%H%M%S").log &
wait
