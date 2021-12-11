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
----------------------------------------
Check data type: val
Load model from: /home/xwx/model-doctor-xwx/output/model/gc/resnet50-20211208-101731
Data size: 10000
----------------------------------------
Test Loss: 0.2462 Acc: 0.9489
Accuracy of  0 : 96.20%
Accuracy of  1 : 97.80%
Accuracy of  2 : 92.60%
Accuracy of  3 : 89.20%
Accuracy of  4 : 96.50%
Accuracy of  5 : 90.70%
Accuracy of  6 : 96.60%
Accuracy of  7 : 96.20%
Accuracy of  8 : 95.90%
Accuracy of  9 : 97.20%
```

PID 6885
resnet50+cifar10
- 从头训练一个模型：0.9489
- 原有模型医生调整：0.9527, + 0.3800%
