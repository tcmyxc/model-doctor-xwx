import numpy as np

label_list = [8, 9]

# for label in label_list:
#     mask_path_pattern = "/nfs/xwx/model-doctor-xwx/modify_kernel/kernel_dict_label_{}.npy".format(label)
#     modify_dict = np.load(mask_path_pattern, allow_pickle=True).item()
#     print(label, ":", modify_dict)
mask_path_label_8 = "/nfs/xwx/model-doctor-xwx/modify_kernel/kernel_dict_label_8.npy"
mask_path_label_9 = "/nfs/xwx/model-doctor-xwx/modify_kernel/kernel_dict_label_9.npy"

a = np.load(mask_path_label_8, allow_pickle=True).item()
b = np.load(mask_path_label_9, allow_pickle=True).item()

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
np.save("kernel_dict_label_89.npy", b)