#!/bin/bash

python3 fine_tuning_v3.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.5 \
    > 三阶段微调-cifar-100-lt-ir100-th0.5.log &
wait

python3 fine_tuning_v3.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.4 \
    > 三阶段微调-cifar-100-lt-ir100-th0.4.log &
wait

python3 fine_tuning_v3.py \
    --data_name cifar-100-lt-ir100 \
    --threshold 0.6 \
    > 三阶段微调-cifar-100-lt-ir100-th0.6.log &
wait
