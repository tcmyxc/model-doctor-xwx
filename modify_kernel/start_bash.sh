#!/bin/bash
python3 fine_tuning.py \
    --lr 1e-2 \
    > 三阶段微调-lr1e-2-最后十层.log &
wait

python3 fine_tuning.py \
    --lr 1e-3 \
    > 三阶段微调-lr1e-3-最后十层.log &
wait


