#!/bin/bash

# python3 /nfs/xwx/model-doctor-xwx/modify_kernel/只训练分类头.py \
#     --lr 1e-3 \
#     > $(date "+%Y%m%d-%H%M%S").log &
# wait

# python3 /nfs/xwx/model-doctor-xwx/modify_kernel/只训练分类头.py \
#     --lr 1e-2 \
#     > $(date "+%Y%m%d-%H%M%S").log &
# wait

# python3 /nfs/xwx/model-doctor-xwx/modify_kernel/只训练分类头.py \
#     --lr 0.05 \
#     > $(date "+%Y%m%d-%H%M%S").log &
# wait

python3 /nfs/xwx/model-doctor-xwx/modify_kernel/只训练分类头.py \
    --lr 0.01 \
    --lr_scheduler custom \
    --loss_type ce \
    > $(date "+%Y%m%d-%H%M%S").log &
wait

python3 /nfs/xwx/model-doctor-xwx/modify_kernel/只训练分类头.py \
    --lr 0.01 \
    --lr_scheduler custom \
    --loss_type fl \
    > $(date "+%Y%m%d-%H%M%S").log &
wait

python3 /nfs/xwx/model-doctor-xwx/modify_kernel/只训练分类头.py \
    --lr 0.01 \
    --lr_scheduler custom \
    --loss_type refl \
    > $(date "+%Y%m%d-%H%M%S").log &
wait