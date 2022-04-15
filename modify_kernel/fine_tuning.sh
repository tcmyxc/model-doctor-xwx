#!/bin/bash

# 只改损失函数
python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --loss_type fl \
    > $(date "+%Y%m%d-%H%M%S").log &
wait

python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --loss_type refl \
    > $(date "+%Y%m%d-%H%M%S").log &
wait

# 修改学习率调度器
python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --lr_scheduler custom \
    > $(date "+%Y%m%d-%H%M%S").log &
wait

# 学习率调度器+损失函数
python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --lr_scheduler custom \
    --loss_type fl \
    > $(date "+%Y%m%d-%H%M%S").log &
wait

python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --lr_scheduler custom \
    --loss_type refl \
    > $(date "+%Y%m%d-%H%M%S").log &
wait
