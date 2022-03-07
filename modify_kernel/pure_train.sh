#!/bin/bash

python3 ./pure_train.py \
    --data_name inaturalist2018 \
    --num_classes 8142 \
    --model_name resnet50 \
    --threshold 0.5 \
    > inaturalist2018-resnet50-th0.5.log &
