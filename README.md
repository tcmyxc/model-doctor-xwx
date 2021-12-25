# 从头训练一个模型，resnet50+cifar-10

```tex
bets res is 194 epoch, optimizer is SGD (
Parameter Group 0
    dampening: 0
    initial_lr: 0.01
    lr: 2.2190176984600013e-05
    momentum: 0.9
    nesterov: False
    weight_decay: 0.0005
)
```

导出依赖包
```bash
conda env export >  model-doctor-conf.yml
```

resnet50+cifar10
1. 从头训练一个模型: 0.9489
2. 原有的模型医生调整: 0.9548, + 0.5900%
3. 只使用正梯度(grad_sift文件用的正梯度，grad_constraint用的绝对值): 0.9562, + 0.7300%
4. 只使用正梯度(grad_sift文件用的正梯度，grad_constraint用的正梯度): 0.9552, + 0.6300%
5. 只使用正激活部分对应的梯度(生成 mask 和训练都这样做): 0.9600, + 1.1100%（batch=16)，0.9542, + 0.5300%(batch=128)
6. 用正激活代替梯度筛选mask，但是训练的时候，使用原有的梯度，正梯度正常约束，负梯度全部约束: 0.9536, + 0.4700%
7. 正激活和正梯度生成的mask取交集，训练同6: 0.9548, + 0.5900%
8. 正激活和正梯度生成的mask取并集，训练同6: 0.9533, + 0.4400%
9. 用正激活部分对应的梯度筛选mask，训练的时候采用原有md方法: 0.9550, + 0.6100%，
    - 0.9575, + 0.8600%（batchsize=64)
    - 0.9600, + 1.1100%（batchsize=32）
10. 用正激活部分对应的梯度筛选mask，训练同6: 0.9545, + 0.5600%
11. 使用正梯度生成mask channel,训练同6：0.9540, + 0.5100%
12. 用正激活代替梯度筛选mask，训练的时候采用原有md方法: 0.9557, + 0.6800%
13. 同上，lr=0.005: 0.9526, + 0.3700%
14. 同上，lr=1e-4: 0.9482, + -0.0700%
15. 原始md，lr=1e-4: 0.9484, + -0.0500%

- 从头训练: acc is 0.948300(prune)


vgg16+stl10
- 从头训练: 0.7094


vgg16+mini-imagenet
- 从头训练：0.7703

alexnet+stl10
- 从头训练：0.6584
- 1倍channel loss：0.6185
