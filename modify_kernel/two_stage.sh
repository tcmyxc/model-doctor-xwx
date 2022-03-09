#!/bin/bash

python3 cbs_refl.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.4 \
    > 二阶段微调-cifar-100-lt-ir100-th0.4.log &
wait

python3 cbs_refl.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.5 \
    > 二阶段微调-cifar-100-lt-ir100-th0.5.log &
wait

python3 cbs_refl.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.6 \
    > 二阶段微调-cifar-100-lt-ir100-th0.6.log &
wait

python3 cbs_refl.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.7 \
    > 二阶段微调-cifar-100-lt-ir100-th0.7.log &
wait

python3 cbs_refl.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.75 \
    > 二阶段微调-cifar-100-lt-ir100-th0.75.log &
wait