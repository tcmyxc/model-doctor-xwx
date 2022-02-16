# 环境相关
```bash
# 导出依赖包
conda env export >  model-doctor-conf.yml

# 从依赖文件重建环境
conda env create -f model-doctor-conf.yml
```

# 记得每次训练修改保存结果目录

# 使用模型医生微调FC层步骤
 1. 修改 `models` 文件夹下 `__init__.py` 文件 `load_modules` 函数
    - `module_modules` 字典，`-1`这个 key 对应层数修改成倒数第二层 FC 层
 2. 使用 `core` 文件夹下 `grad_sift_fc.py` 文件生成对应的 `channel mask`
 3. 修改 `core` 文件夹下 `grad_constraint.py` 文件
    - 注释第 39 行
    - 取消第 40 行的注释
4. 使用以前的逻辑微调模型

# 添加数据集步骤
1. 准备数据集
2. 参考 `loaders` 文件夹的数据加载器，自己写一个数据加载器
3. 在 `loaders` 文件夹的 `__init__.py` 文件注册自己的数据加载器，需要改两个地方
    - `load_data` 函数里面 `if` 判断两种情况都要加上


# 常规数据集训练
## 从头训练一个模型，resnet50+cifar-10

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



## resnet50+cifar10
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

# 验证模型医生应用到FC层是否有效

如果使用模型医生, vgg16 的 loss_channel 的放缩比应该是1，res50应该是10
alexnet 只有5层卷积
验证模型医生对卷积层有效，可用的模型：alexnet，vgg16，senet34

## vgg16+stl10
- 从头训练: 0.7094
- 使用模型医生调整倒数第二层 FC: 0.7374
- 倒数第一层卷积层: 0.7375


## vgg16+mini-imagenet
- 从头训练：0.7703


## senet34+stl10
- 预训练模型：0.8200


## alexnet+stl10
- 师兄用的正数第二层卷积，我这里用的倒数第一层卷积
- 从头训练：0.6584
- 1倍channel loss：78轮loss开始变成nan
- 0.1倍loss：116轮开始分类的loss开始变成恒定2左右


## alexnetv2+stl10
- 使用pytorch官方给的模型结构从头训练：0.6985
- 1倍channel loss, 原始的模型医生：0.7258
- 1倍channel loss, 倒数第二层 FC 层：0.7061


## alexnetv2+cifar10(图片resize成224x224)
- 模型结果从pytorch官方直接拷贝得到的
- 预训练模型：0.9036
- 10倍channel loss，模型医生：0.9052
- 10倍channel loss，FC+模型医生：0.9014


## alexnetv3+cifar10（图片大小32x32）
- 这个模型改了alexnetv2第一层卷积层的参数，最后一层池化使用自适应池化代替
- 预训练模型：0.8513
- 10倍channel loss，准确率在 10%，一直上不去，原始分类的loss在 2 左右不变，如果去除添加的噪声，分类的loss变成0.25左右
猜想师兄用的 torch.randn 对特征图改变较大（猜想正确，去除噪声能正常训练起来了），但是我没跑最终的实验
- 1倍channel loss: 0.8555


## alexnetv2+cifar10(图片resize成64x64)
- 预训练模型：0.8389

- 10倍channel loss, 倒数第二层 FC 层：0.8408
- 1倍channel loss, 倒数第二层 FC 层：0.8326

- 1倍channel loss, 倒数第一层卷积, 原始模型医生: 训练不起来
- 1倍channel loss, 倒数第一层卷积, 模型医生不加噪音，可以训练起来：0.8347

- 10倍channel loss, 倒数第一层卷积, 原始模型医生：0.8111(训练了400轮，train还没到100%)
    - 有时候一开始训练loss就nan了，可以kill掉重新开始
- 10倍channel loss, 倒数第一层卷积, 模型医生不加噪音：

# 长尾数据集+模型医生
## cifar-10(ρ=100)类别分布
| 类别 | 数量 |
| ---  | --- |
| 0 | 5000 |
| 1 | 2997 |
| 2 | 1796 |
| 3 | 1077 |
| 4 | 645 |
| 5 | 387 |
| 6 | 232 |
| 7 | 139 |
| 8 | 83 |
| 0 | 50 |



## resnet50+cifar-100(ρ=100)
- 预训练：0.3770
- 模型医生：0.3792

## resnet50+cifar-10(ρ=100)
- 预训练：0.7091
    ```tex
    acc of  0 : 94.60%
    acc of  1 : 94.20%
    acc of  2 : 74.50%
    acc of  3 : 65.40%
    acc of  4 : 81.00%
    acc of  5 : 60.30%
    acc of  6 : 84.20%
    acc of  7 : 55.50%
    acc of  8 : 40.50%
    acc of  9 : 58.90%
    ```
- 模型医生：0.7198
    ```tex
    acc of  0 : 97.10%
    acc of  1 : 98.50%
    acc of  2 : 79.90%
    acc of  3 : 78.30%
    acc of  4 : 81.20%
    acc of  5 : 63.80%
    acc of  6 : 72.00%
    acc of  7 : 57.50%
    acc of  8 : 33.90%
    acc of  9 : 57.60%
    ```


## resnet50+cifar-10(ρ=10)
- 预训练：0.8848
    ```tex
        acc of  0 : 96.70%
        acc of  1 : 98.10%
        acc of  2 : 89.30%
        acc of  3 : 82.10%
        acc of  4 : 90.20%
        acc of  5 : 79.70%
        acc of  6 : 90.60%
        acc of  7 : 85.80%
        acc of  8 : 86.10%
        acc of  9 : 86.20%
    ```
- 模型医生：0.8958
    ```tex
    acc of  0 : 97.00%
    acc of  1 : 98.50%
    acc of  2 : 91.50%
    acc of  3 : 85.10%
    acc of  4 : 90.50%
    acc of  5 : 82.50%
    acc of  6 : 89.70%
    acc of  7 : 85.80%
    acc of  8 : 88.70%
    acc of  9 : 86.50%
    ```

## resnet50+cifar-10(ρ=50)
- 预训练：0.7744
    ```tex
        acc of  0 : 92.40%
        acc of  1 : 96.90%
        acc of  2 : 76.20%
        acc of  3 : 71.70%
        acc of  4 : 82.80%
        acc of  5 : 70.10%
        acc of  6 : 77.70%
        acc of  7 : 78.00%
        acc of  8 : 69.70%
        acc of  9 : 58.90%
    ```
- 模型医生：0.7753
    ```tex
    acc of  0 : 96.50%
    acc of  1 : 99.00%
    acc of  2 : 86.90%
    acc of  3 : 79.10%
    acc of  4 : 82.40%
    acc of  5 : 70.00%
    acc of  6 : 79.70%
    acc of  7 : 68.10%
    acc of  8 : 60.90%
    acc of  9 : 52.70%
    ```

# 验证 iNaturalist 2018 数据集
- `md5sum 2018.tar.gz` should produce `b1c6952ce38f31868cc50ea72d066cc3`

# ResNext50+ImageNet-lt
- 预训练：acc1 is 40.45%, acc5 is 65.85%


# resnet50 + cifar-10-lt-ir100
- 微调阶段只使用FL
    ```tex
    acc of  0 : 89.50%
    acc of  1 : 96.20%
    acc of  2 : 76.20%
    acc of  3 : 75.70%
    acc of  4 : 85.50%
    acc of  5 : 61.80%
    acc of  6 : 75.10%
    acc of  7 : 62.00%
    acc of  8 : 52.50%
    acc of  9 : 61.50%

    [val] Loss CLS: 0.9257 Loss GC: 0.0000 Acc: 0.7360

    [val] acc1 is 73.60%, acc5 is 97.94%
    ```


# ResNet32+Cifar-10-lt-ir100（直接训练）
- CE loss：acc1 is `69.34%`, err rate is `30.66%`, CB论文可以做到 err `29.64%`
- REFL
    - th=0.5: acc1 is 68.69%
    - th=0.4: acc1 is **70.72%**, err rate is **29.28%**; acc1 is **71.82%**, err1 is **28.18%**（自定义学习率）
    - th=0.3: acc1 is 70.25%
    - th=0.25: acc1 is 66.55%
- REFL V2
    - th=0.5: acc1 is 65.66%
    - th=0.4: acc1 is 67.53%
- REFL V3
    - th=0.4: acc1 is 69.61%
- REFL V4:
    - th=0.4: acc1 is 68.55%
- FL: acc1 is 69.20%
- EFL: acc1 is 67.79%
- DFL: acc1 is **70.71%**
- RFL: acc1 is 68.66%


# ResNet32+Cifar-100-lt-ir100（直接训练）
- CE loss: acc1 is 40.58%, acc5 is 69.95%, err1 is **59.42%**, err5 is 30.05%
- FL: acc1 is 39.11%, acc5 is 69.10%, err1 is **60.89%**, err5 is 30.90%
- REFL
   - th=0.4: acc1 is 36.97%, acc5 is 67.70%
   - 同上，自定义学习率：acc1 is 39.73%, acc5 is 68.50%, err1 is **60.27%**, err5 is 31.50%