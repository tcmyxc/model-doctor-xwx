#!/bin/bash

# python3 /mnt/nfs/xwx/model-doctor-xwx/modify_kernel/修改最后一层特征图.py \
#     --data_name cifar-10-lt-ir100 \
#     --lr 0.1 \
#     --lr_scheduler custom \
#     --loss_type bsl \
#     --gpu_id 1 \
#     --epochs 200 \
#     > $(date "+%Y%m%d-%H%M%S").log
# wait

# python3 /mnt/nfs/xwx/model-doctor-xwx/modify_kernel/修改最后一层特征图.py \
#     --data_name cifar-10-lt-ir100 \
#     --lr 0.01 \
#     --lr_scheduler custom \
#     --loss_type bsl \
#     --gpu_id 1 \
#     --epochs 200 \
#     > $(date "+%Y%m%d-%H%M%S").log
# wait

# python3 /mnt/nfs/xwx/model-doctor-xwx/modify_kernel/修改最后一层特征图.py \
#     --data_name cifar-10-lt-ir100 \
#     --lr 0.001 \
#     --lr_scheduler custom \
#     --loss_type bsl \
#     --gpu_id 1 \
#     --epochs 200 \
#     > $(date "+%Y%m%d-%H%M%S").log
# wait

# python3 /mnt/nfs/xwx/model-doctor-xwx/modify_kernel/修改最后一层特征图.py \
#     --data_name cifar-10-lt-ir100 \
#     --lr 0.1 \
#     --loss_type bsl \
#     --gpu_id 1 \
#     --epochs 200 \
#     > $(date "+%Y%m%d-%H%M%S").log
# wait

python3 /mnt/nfs/xwx/model-doctor-xwx/modify_kernel/修改最后一层特征图.py \
    --data_name cifar-10-lt-ir100 \
    --lr 0.01 \
    --loss_type ce \
    --gpu_id 1 \
    --epochs 40 \
    > $(date "+%Y%m%d-%H%M%S").log
wait


# python3 /mnt/nfs/xwx/model-doctor-xwx/modify_kernel/修改最后一层特征图.py \
#     --data_name cifar-10-lt-ir100 \
#     --lr 0.001 \
#     --loss_type bsl \
#     --gpu_id 1 \
#     --epochs 200 \
#     > $(date "+%Y%m%d-%H%M%S").log
# wait
