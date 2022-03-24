#!/bin/bash

python3 ./one_stage.py \
    --data_name cifar-100-lt-ir100 \
    --num_classes 100 \
    --model_name resnet32 \
    --threshold 0.4 \
    > cifar-100-lt-ir100-resnet32-th0.4.log &
wait

python3 ./one_stage.py \
    --data_name cifar-100-lt-ir50 \
    --num_classes 100 \
    --model_name resnet32 \
    --threshold 0.4 \
    > cifar-100-lt-ir50-resnet32-th0.4.log &
wait

python3 ./one_stage.py \
    --data_name cifar-100-lt-ir10 \
    --num_classes 100 \
    --model_name resnet32 \
    --threshold 0.4 \
    > cifar-100-lt-ir10-resnet32-th0.4.log &
wait
