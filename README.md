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
| 9 | 50 |


# ResNext50+ImageNet-lt
- 预训练：acc1 is 40.45%, acc5 is 65.85%

# ResNet32+Cifar-100-lt-ir100（直接训练）
- CE loss: acc1 is 40.58%, acc5 is 69.95%, err1 is **59.42%**, err5 is 30.05%
- FL: acc1 is 39.11%, acc5 is 69.10%, err1 is **60.89%**, err5 is 30.90%
- REFL
    - th=0.5: acc1 is 41.77%, acc5 is 71.12%, err1 is **58.23%**,
    - th=0.4: acc1 is 36.97%, acc5 is 67.70%;（自定义学习率）: acc1 is 39.73%, acc5 is 68.50%, err1 is **60.27%**, err5 is 31.50%
    - th=0.3: acc1 is 36.65%, acc5 is 61.91%
- EFL: acc1 is 38.94%, acc5 is 68.23%, err1 is 61.06%, err5 is 31.77%
- REFL+MD:
    - th=0.2, 初始学习率0.01: acc1 is 41.60%
    - th=0.2, 初始学习率0.001: acc1 is 42.48%
    - th=0.1, 初始学习率0.001: acc1 is 41.64%, acc5 is 71.24%

# 2022年3月3日

目前实验的实验步骤:
1. 使用普通的数据加载器和模型训练，loss函数使用REFL，使用自定义的学习率调度器
2. 使用步骤1训练好的模型再次在**训练集**上面跑一遍（普通的数据加载器），挑选高置信的图片
    - 使用 `image_sift.py` 筛选图片
3. 拿着步骤2得到的图片，使用 `pattern_sift.py` 算每个类别的高于平均梯度的卷积核
4. 使用 `读取npy.py` 合并每个类别有关的卷积核
5. 使用 kernel_dict 文件夹下面的脚本训练模型

# ResNet32+Cifar-10-lt-ir100（直接训练）
- CE loss，**baseline**: acc1 is `69.34%`, err rate is `30.66%`, CB论文可以做到 err `29.64%`
- REFL
    - th=0.5: acc1 is 68.69%
    - th=0.4: acc1 is 70.72%, err rate is 29.28%; acc1 is **71.82%**, err1 is **28.18%**（自定义学习率）, 后续的SOTA模型
    - th=0.3: acc1 is 70.25%
    - th=0.25: acc1 is 66.55%
- REFL+CBS
    - th=0.5
        - lr=1e-2: 70.99%
        - lr=1e-3: 59.61%
- REFL V2
    - th=0.5: acc1 is 65.66%
    - th=0.4: acc1 is 67.53%
- REFL V3
    - th=0.4: acc1 is 69.61%
- REFL V4:
    - th=0.4: acc1 is 68.55%
- FL: acc1 is 69.20%
- EFL: acc1 is 67.79%
- DFL: acc1 is 70.71%
- RFL: acc1 is 68.66%
- REFL+MD（初始学习率0.01）
    - th=0.5: acc1 is 73.21%
    - th=0.4: acc1 is 74.14%, (余弦, init_lr=0.01): acc1 is 72.97%
    - th=0.3: acc1 is 74.76%
    - th=0.25: acc1 is **74.92%**
    - th=0.2: acc1 is 73.09%


- 以下默认使用refl，挑选卷积核使用**测试集**高置信度图片，除非另加说明(数据泄露)
- baseline: 71.82%
- 挑选卷积核(最后一层卷积核)
    - 自定义学习率: 72.82%
    - 只选最后一个尾部类
        - 余弦退火, th=0.5:  **73.11%**
        - 余弦退火, th=0.4:  72.85%
        - 余弦退火, th=0.25: 72.98%
    - 选最后2个尾部类
        - 余弦退火, th=0.5:  72.53%
    - 选最后3个尾部类
        - 余弦退火, th=0.5:  72.67%
    - 选5、7、9类
        - 余弦退火, th=0.5:  72.48%
- 挑选卷积核（所有层）
    -  只选最后一个尾部类
        - 余弦退火, th=0.5:  72.98%
- 挑选卷积核（最后10层）
    - 只选最后一个尾部类
        - 余弦退火, th=0.5:   **73.10%**
        - 自定义学习率, th=0.5: 73.05%
    - 选最后两个类
        - 余弦退火, th=0.5:     72.66%
        - 自定义学习率, th=0.5:  72.62%
        - 余弦退火, th=0.5, fl: 72.97%
        - 自定义学习率, th=0.5, fl: **73.40%**

- 下面的使用**训练集**高置信度图片挑选卷积核, 默认使用refl
- baseline: 71.82%
- 挑选卷积核（最后10层）
    - 选最后两个类
        - 自定义学习率, th=0.5, fl: 72.93%
        - 自定义学习率, th=0.5: **73.94%**
    - 后3个类
        - 余弦退火,th=0.5: 72.43%
        - 自定义学习率, th=0.5: 72.55%
    - **选的尾部类太多会掉点**
- 上面的代码有误，导致前20层正常更新，后10层只更新了部分卷积核
- 挑选卷积核（所有层）
    - 选最后两个类
        - 自定义学习率, th=0.5: 72.65%
        - 余弦退火,th=0.5: 72.62%
- 挑选卷积核（最后10层，前面20层不进行梯度更新）
    - 后3个类
        - 自定义学习率, th=0.5: 72.36%
- 挑选卷积核（使用对应类别图片的loss更新对应的卷积核，未加说明th=0.5）
    - lr=0.01
        - 自定义学习率，所有类别，后10层: 65.49%
        - 自定义学习率，所有类别，所有层: 63.29%
        - 同上，但是反向调整卷积核: 66.06%
        - (消融实验)自定义学习率，类别平衡采样(CBS)，REFL，不更新卷积核: **77.20%**, Epoch 2
    - lr=1e-4
        - 自定义学习率，所有类别，所有层: 74.14%, Epoch 8
        - 同上，但是反向调整卷积核: 74.16%, Epoch 7
        - 余弦退火，其余同1: 74.81%, Epoch 2
    - lr=1e-5
        - 自定义学习率，所有类别，所有层: **75.77%**, Epoch 26
        - 余弦退火，其余同1: 75.33%, Epoch 26
        - 自定义学习率，所有类别，所有层，th=0.4: 74.11%, Epoch 14
        - 自定义学习率，所有类别，所有层，th=0.6: 75.05%, Epoch 38
        - 使用类别均衡采样策略，其余同1: 75.55%, Epoch 9(中途服务器挂了，只跑了165轮); 74.79%, Epoch 10
    - lr=1e-6
        - 自定义学习率，所有类别，所有层: **75.90%**, Epoch 179
        - 自定义学习率, 所有类别, 所有层, th=0.4: 75.57%, Epoch 145
        - 使用类别均衡采样策略，其余同1: 75.89%, Epoch 75
    


尝试了使用特定类别的图片
1. 更新对应的卷积核（没啥用）
2. 反向更新卷积核（没啥用）
3. 只手动更新后十层，其余层自行更新（没啥用）
4. 调整学习率为1e-4，有效果
5. 调整学习率为1e-5，效果提升显著，提升了 2 个点
6. 不调整卷积核，仅仅在预训练模型上面加了CBS和REFL，不调整卷积核，提升了 6 个点，结果比以前都要好（好过调整卷积核的实验）


训练集准确率低的原因：同一批数据，不同类别的图片进去的时候模型状态已经改变了

```bash
git config --global user.name tcmyxc
git config --global user.email 1282494272@qq.com
```

```bash
$(date "+%Y%m%d-%H%M%S").log
```


# ResNet32+Cifar-10-lt-ir100
- 预训练：71.82%(refl), (ce:69.34%, fl:69.20%)

- 在预训练的基础上，重新训练，配置同预训练，不加额外损失
    - lr0.1, refl, custom: 72.86%（首轮即终点）
    - lr0.1, ce, custom:   72.06%（首轮即终点）
    - lr0.1, fl, custom:   71.82%

- 分类错误样本，不相关卷积核的特征图向聚类中心靠近(所有配置同预训练,加载预训练模型)
    - lr0.1, refl, custom: 69.56%
    - lr0.1, ce, custom:   70.91%
    - lr0.1, fl, custom:   69.73%
- 分类错误样本，不相关卷积核的特征图向聚类中心靠近(不加载预训练模型)
    - lr0.1, refl, custom: 66.37%
    - lr0.1, ce, custom:   63.08%
    - lr0.1, fl, custom:   65.70%


- 分类错误样本的特征图向聚类中心靠近(加载预训练模型)
    - lr0.1, refl, custom: 72.25%
    - lr0.1, ce, custom:   72.48%
    - lr0.1, fl, custom:   72.41%（首轮即终点）
- 分类错误样本的特征图向聚类中心靠近(不加载预训练模型)
    - lr0.1, refl, custom: 74.22%
    - lr0.1, ce, custom:   72.52%
    - lr0.1, fl, custom:   72.18%
- 分类错误样本的特征图向聚类中心靠近(加载预训练模型)
    - lr1e-3, refl, custom: 72.46%(cosine, 72.52%)
    - lr1e-3, ce, custom:   72.19%(cosine, 72.00%)
    - lr1e-3, fl, custom:   72.74%(cosine, 72.73%)

- 如果本次batch中存在尾部样本，那就只更新在kernel_tail中的卷积核；如果不存在尾部样本，就正常更新
    - lr1e-3, cosine
        - 后2个类+ce:   72.26%
        - 后3个类+ce:   72.24%
        - 后2个类+fl:   **73.06%**
        - 后2个类+refl: 72.69%

- 只更新在kernel_tail中的卷积核
    - lr1e-3, cosine
        - ce:   72.59%
        - fl:   **73.12%**
        - refl: 72.78%
    - 使用和预训练一样的配置
        - ce:   72.74%
        - fl:   72.82%
        - refl: 72.88%
    - lr1e-2, cosine
        - ce:   72.84%
        - fl:   72.60%
        - refl: 72.52%
    - 更新的用力一点（10倍梯度）
        - lr1e-3, cosine: 72.52%
        - 训练策略同预训练：72.79%


- 在所有的卷积层都更新那些对所有类别贡献度都低的卷积核
    - lr1e-3, cosine
        - ce: 71.87%

- 在最后10层卷积层都更新那些对所有类别贡献度都低的卷积核(lr1e-3, cosine, ce)
    - 阈值1：71.84%
    - 阈值2：71.97%
    - 阈值2 + cbs：77.74%(77.71%, 77.71%, 77.79%)
    - 阈值3：72.12%
    - 阈值5：72.51%
    - (消融实验)cbs：77.66%(77.76%, 77.58%, 77.64%)

- 长尾数据集的尾部类别模仿平衡数据集的卷积核占用
    - 类别8, 9: 72.26%
    - 类别9: 72.54%

- HC LOSS
    - lr1e-3, cosine
        - ce: 73.58%, 72.76%
        - fl: 73.58%
        - refl: 73.05%
    - lr1e-2, cosine
        - ce: 71.48%
        - fl: 73.54%
        - refl: 72.19%

- hc 调整平衡数据集
    - 预训练：92.07%
    - lr1e-3, cosine
        - ce+hc: 92.20%
        - fl+hc: 92.18%
    
    - lr1e-2, cosine
        - ce+hc: 91.52%
        - fl+hc: 90.88%

- 只调整分类头（200轮）
    - lr0.1： 71.95%
    - lr0.05: 71.64%
    - lr0.01: 71.44%
    - lr1e-3: 67.98%

- 只调整分类头，自定义学习率（200轮）
    - lr0.1
        - ce:  72.03%
        - fl:  72.72%
        - refl: 72.70%

- 只调整分类头，自定义学习率，同时附加hcl（200轮）
    - lr0.1
        - ce:  75.09% (可视化个权重)
        - fl:  74.89%
        - refl: 75.11%
    - lr0.01
        - ce:  73.66%
        - fl:  73.90%
        - refl: 73.76%

只训练分类头，使用正则化方法
```python
self.weight = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
```

```
- 修改头部类相关卷积核的特诊图，附加hcl
    - 预训练: 71.82
    - lr1e-2, cosine, ce: 70.08 (epoch 125)
    - lr1e-3, cosine, ce: 76.04 (epoch 78)
    
    - lr1e-2, custom, ce: 72.67 (epoch 1)
    - lr1e-3, custom, ce: 75.41 (epoch 33)

- 修改头部类相关卷积核的特诊图
    - 预训练: 71.82
    
    - lr1e-1, cosine, ce: 74.19 (epoch 179)
    - lr1e-2, cosine, ce: 75.51 (epoch 71)
    - lr1e-3, cosine, ce: 74.52 (epoch 163)

    - lr1e-1, cosine, refl: 71.95 (epoch 179)
    - lr1e-2, cosine, refl: 75.13 (epoch 88)
    - lr1e-3, cosine, refl: 74.52 (epoch 163)
    
    - lr1e-1, custom, ce: 74.02 (epoch 3)
    - lr1e-2, custom, ce: 76.30 (epoch 119)
    - lr1e-3, custom, ce: 74.66 (epoch 141)

    - lr1e-1, custom, refl: 74.40 (epoch 3)
    - lr1e-2, custom, refl: 75.36 (epoch 134)
    - lr1e-3, custom, refl: 74.43 (epoch 163)
```


5.在最后10层卷积层都更新那些对所有类别贡献度都低的卷积核
- (1)阈值1：71.84%，+0.02%
- (2)阈值2：71.97%，+0.15%
- (3)阈值3：72.12%，+0.30%
- (4)阈值5：72.51%，+0.69%


# ResNet32+Cifar-100-lt-ir100
- 预训练：41.77%(REFL), (CE:40.58%, FL:39.11%)


# 2022年7月的预训练模型

- cifar-10-lt-ir10
    - lr1e-1, custom
        - ce: 87.49
            - lr1e-3, custom, ce: 88.69 (+1.2) 
            - lr1e-2, custom, ce: 88.76 (+1.27) 
        - fl: 86.34
        - refl: 86.80

- cifar-10-lt-ir100
    - lr1e-1, custom
        - ce: 72.94
            - lr1e-3, custom, ce: 74.79
            - lr1e-2, custom, ce: 75.67
        - fl: 71.37
        - refl: 73.43 


- cifar100修改前30%基本上没效果
- cifar-100-lt-ir10
    - lr1e-1, custom
        - ce: 59.22 (58.10)
            - 修改前 5%
                - lr1e-3, custom, ce: 59.25
                - lr1e-2, custom, ce: 58.89
            - 修改前 10%
                - lr1e-3, custom, ce: 59.27 (58.23)
                - lr1e-2, custom, ce: 58.70 (57.78)
            - 修改前 30%
                - lr1e-2, custom, ce: (56.94)
                - lr1e-3, custom, ce: (57.62)
        - fl: 57.70
        - refl: 58.52


- cifar-100-lt-ir50
    - lr1e-1, custom
        - ce: 45.47 (45.24)
            - 修改前5%
                - lr1e-3, custom, ce: 46.02
                - lr1e-2, custom, ce: 45.92
            - 修改前 10%
                - lr1e-3, custom, ce: 46.34 (45.98)
                - lr1e-2, custom, ce: 46.24 (45.88)
            - 修改前 30%
                - lr1e-2, custom, ce: (46.57)
                - lr1e-3, custom, ce: (46.52)
        - fl: 44.47
        - refl: 44.77

- cifar-100-lt-ir100
    - lr1e-1, custom
        - ce: 39.12 (40.35)
            - 修改前5%
                - lr1e-3, custom, ce: 39.97
                - lr1e-2, custom, ce: 40.03
            - 修改前 10%
                - lr1e-3, custom, ce: 40.64 (41.91)
                - lr1e-2, custom, ce: 40.49 (42.06)
            - 修改前 30%
                - lr1e-2, custom, ce: (43.16)
                - lr1e-3, custom, ce: (42.63)
        - fl: 40.59
        - refl: 39.19

# 2022年7月11日的预训练模型

- cifar-100-lt-ir10
    - lr1e-1, custom
        - ce: 59.03
            - 修改前 30%
                - lr1e-2, custom, ce: 57.77
                - lr1e-3, custom, ce: 
        - fl: 
        - refl: 


- cifar-100-lt-ir50
    - lr1e-1, custom
        - ce: 45.06
            - 修改前 30%
                - lr1e-2, custom, ce: 46.36
                - lr1e-3, custom, ce: 
        - fl: 
        - refl: 

- cifar-100-lt-ir100
    - lr1e-1, custom
        - ce: 40.96
            - 修改前 30%
                - lr1e-2
                    - custom, ce: 41.88
                    - cosine, ce, 只训练分类头
                        - 10 epoch: 40.83

                - lr1e-3
                    - custom, ce: 
                - lr1e-1
                    - cosine, ce, 只训练分类头
                        - 10 epoch: 41.09 (epoch 1)
        - fl: 
        - refl: 


# 训练更强的baseline

- cifar-10-lt-ir100
    - lr1e-1, custom
        - bsl:  77.28
            - lr1e-1, custom, bsl: 76.18 (epoch 1)
            - lr1e-2, custom, bsl: 77.83
            - lr1e-3, custom, bsl: 77.04 (epoch 1)

            - lr1e-1, cosine, bsl, all:
                - epoch 10: 78.26, 78.26, 78.26

            - lr1e-1, cosine, bsl, linear: 
                - epoch 10: 78.04, 78.04, 78.04

            - lr1e-1, cosine, bsl, linear+layer3: 
                - epoch 10:  78.61, 78.61, 78.61
                - epoch 20:  78.77, 78.77
                - epoch 30:  78.85
                - epoch 40:  79.11
                - epoch 50:  79.19
                - epoch 60:  78.96
                - epoch 70:  78.65
                - epoch 80:  78.36
                - epoch 90:  78.69
                - epoch 100: 78.22
                - epoch 110: 78.91
                - epoch 120: 78.02
                - epoch 130: 77.59
                - epoch 140: 78.13
                - epoch 150: 77.94
                - epoch 160: 77.89
                - epoch 170: 77.75
                - epoch 180: 78.27
                - epoch 190: 77.83
                - epoch 200: 77.33
        
        - cbl:  73.95
        - ce:   70.46
        - fl:   73.50
        - refl: 70.99

        - bsl+auto_aug: 82.45
            - lr1e-1, cosine, bsl, linear+layer3: 
                - epoch 10: 74.93 (一路下降)
            
            - lr1e-1, custom, bsl: 80.76 (epoch 1)
            - lr1e-2, custom, bsl: 82.84
            - lr1e-3, custom, bsl: 83.18

            - lr1e-1, cosine, bsl: 69.89 (epoch 1)
            - lr1e-2, cosine, bsl: 82.89
            - lr1e-3, cosine, bsl: 83.23

            - 在预训练模型基础上重新训练, 配置不变: 83.88
            - 在预训练模型基础上重新训练, 只修改学习率, 其余配置不变
                - lr1e-2: 83.47