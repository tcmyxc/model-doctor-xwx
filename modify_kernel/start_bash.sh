#!/bin/bash

python3 fine_tuning.py \
    --lr 1e-4 \
    --threshold 0.6 \
    > 三阶段微调-lr1e-4-th0.6-所有层.log &
wait

python3 fine_tuning.py \
    --lr 1e-4 \
    --threshold 0.7 \
    > 三阶段微调-lr1e-4-th0.7-所有层.log &
wait

python3 fine_tuning.py \
    --lr 1e-4 \
    --threshold 0.8 \
    > 三阶段微调-lr1e-4-th0.8-所有层.log &
wait

python3 fine_tuning.py \
    --lr 1e-5 \
    --threshold 0.6 \
    > 三阶段微调-lr1e-5-th0.6-所有层.log &
wait

python3 fine_tuning.py \
    --lr 1e-5 \
    --threshold 0.7 \
    > 三阶段微调-lr1e-5-th0.7-所有层.log &
wait

python3 fine_tuning.py \
    --lr 1e-5 \
    --threshold 0.8 \
    > 三阶段微调-lr1e-5-th0.8-所有层.log &
wait

python3 fine_tuning.py \
    --lr 1e-6 \
    --threshold 0.6 \
    > 三阶段微调-lr1e-6-th0.6-所有层.log &
wait

python3 fine_tuning.py \
    --lr 1e-6 \
    --threshold 0.7 \
    > 三阶段微调-lr1e-6-th0.7-所有层.log &
wait

python3 fine_tuning.py \
    --lr 1e-6 \
    --threshold 0.8 \
    > 三阶段微调-lr1e-6-th0.8-所有层.log &
wait
