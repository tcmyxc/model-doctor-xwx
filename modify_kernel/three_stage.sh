#!/bin/bash

python3 fine_tuning_v3.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.5 \
    --lr 1e-5 \
    > 三阶段微调-cifar-100-lt-ir100-th0.5-lr1e-5.log &
wait

python3 fine_tuning_v3.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.4 \
    --lr 1e-5 \
    > 三阶段微调-cifar-100-lt-ir100-th0.4-lr1e-5.log &
wait

python3 fine_tuning_v3.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.6 \
    --lr 1e-5 \
    > 三阶段微调-cifar-100-lt-ir100-th0.6-lr1e-5.log &
wait

python3 fine_tuning_v3.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.5 \
    --lr 1e-4 \
    > 三阶段微调-cifar-100-lt-ir100-th0.5-lr1e-4.log &
wait

python3 fine_tuning_v3.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.4 \
    --lr 1e-4 \
    > 三阶段微调-cifar-100-lt-ir100-th0.4-lr1e-4.log &
wait

python3 fine_tuning_v3.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.6 \
    --lr 1e-4 \
    > 三阶段微调-cifar-100-lt-ir100-th0.6-lr1e-4.log &
wait
