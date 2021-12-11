#!/bin/bash
export data_name='cifar-10'
#export model_name='alexnet' #0a
#export model_name='vgg16' #1b
#export model_name='resnet34' #2c
export model_name='resnet50' #3d
#export model_name='senet34' #4e
#export model_name='wideresnet28' #5f
#export model_name='resnext50' #6g
#export model_name='densenet121' #7h
#export model_name='simplenetv1' #8i
#export model_name='efficientnetv2s' #9j
#export model_name='efficientnetv2l' #10k
#export model_name='googlenet' #11l
#export model_name='xception' #12m
#export model_name='mobilenetv2' #13n
#export model_name='inceptionv3' #14o
#export model_name='shufflenetv2' #15p
#export model_name='squeezenet' #16q
#export model_name='mnasnet' #17r
export result_name='gc/'${model_name}'-20211208-101731-posgrad'
export pretrained_name=${model_name}'-20211208-101731'
export model_layers='-1'
export device_index='0'
# 导出自己的代码路径到python环境变量
export PYTHONPATH=${PYTHONPATH}:/home/xwx/model-doctor-xwx
python3 engines/train_cls_grad.py --data_name ${data_name} --model_name ${model_name} --result_name ${result_name}  --pretrained_name ${pretrained_name}  --model_layers ${model_layers}  --device_index ${device_index}
# dos2unix scripts/train_cls_grad.sh
# bash scripts/train_cls_grad.sh
