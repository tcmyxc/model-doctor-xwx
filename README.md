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

16. 2021年12月26日
    - 从头训练: acc is 0.9484(最优的val acc)
    - md: 0.9550


如果使用模型医生, vgg16 的 loss_channel 的放缩比应该是1，res50应该是10

vgg16+stl10
- 从头训练: 0.7094
- 使用模型医生调整倒数第二层 FC: 0.7374
- 倒数第一层卷积层: 0.7375



vgg16+mini-imagenet
- 从头训练：0.7703

alexnet+stl10
- 从头训练：0.6584
- 1倍channel loss：78轮loss开始变成nan
- 0.1倍loss：116轮开始分类的loss开始变成恒定2左右
- 使用pytorch官方给的模型结构从头训练：0.6985
- 原始的模型医生：0.7258
- 倒数第二层 FC 层：0.7061

senet34+stl10
- 预训练模型：0.8200
- 倒数第二层 FC 层：


# 使用模型医生微调FC层步骤
 1. 修改 `models` 文件夹下 `__init__.py` 文件 `load_modules` 函数
    - `module_modules` 字典，`-1`这个 key 对应层数修改成倒数第二层 FC 层
 2. 使用 `core` 文件夹下 `grad_sift_fc.py` 文件生成对应的 `channel mask`
 3. 修改 `core` 文件夹下 `grad_constraint.py` 文件
    - 注释第 39 行
    - 取消第 40 行的注释
4. 使用以前的逻辑微调模型
