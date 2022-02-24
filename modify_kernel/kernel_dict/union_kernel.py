import numpy as np
import os

label_list = [5, 6789]

# for label in label_list:
#     mask_path_pattern = "/nfs/xwx/model-doctor-xwx/modify_kernel/kernel_dict_label_{}.npy".format(label)
#     modify_dict = np.load(mask_path_pattern, allow_pickle=True).item()
#     print(label, ":", modify_dict)
mask_path_label_7 = f"/nfs/xwx/model-doctor-xwx/modify_kernel/kernel_dict/kernel_dict_label_{label_list[0]}.npy"
mask_path_label_89 = f"/nfs/xwx/model-doctor-xwx/modify_kernel/kernel_dict/kernel_dict_label_{label_list[1]}.npy"
print(mask_path_label_7)
print(mask_path_label_89)

a = np.load(mask_path_label_7, allow_pickle=True).item()
b = np.load(mask_path_label_89, allow_pickle=True).item()

print("-"*40)
for k, v in a.items():
    print(k, v)

print("-"*40)
for k, v in b.items():
    print(k, v)

for k, a_v in a.items():
        for internal_v in a_v[-1]:
            # print(internal_v)
            if internal_v not in b[k][-1]:
                b[k][-1].append(internal_v)
print("-"*40)
for k, v in b.items():
    print(k, v)

save_root = "/nfs/xwx/model-doctor-xwx/modify_kernel/kernel_dict"
result_name = f"kernel_dict_label_{label_list[0]}{label_list[1]}.npy"
result_name = os.path.join(save_root, result_name)
np.save(result_name, b)