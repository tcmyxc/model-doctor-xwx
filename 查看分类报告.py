import numpy as np

path_to_report = "/nfs/xwx/model-doctor-xwx/output/model/three-stage/resnet32/cifar-100-lt-ir100/lr0.01/th0.5/custom_lr_scheduler/ce_loss/2022-07-06_01-32-37/cls_test_acc_report.npy"

report = np.load(path_to_report, allow_pickle=True)

for k, v in report.item().items():
    print(k, ":", len(v))

