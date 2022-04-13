#!/bin/bash

python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --epochs 500 \
    --loss_type ce \
    > $(date "+%Y%m%d-%H%M%S").log &
wait


python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --epochs 500 \
    --loss_type fl \
    > $(date "+%Y%m%d-%H%M%S").log &
wait

python3 /nfs/xwx/model-doctor-xwx/modify_kernel/fine_tuning_v10.py \
    --epochs 500 \
    --loss_type refl \
    > $(date "+%Y%m%d-%H%M%S").log &
wait

