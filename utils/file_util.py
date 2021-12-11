import sys

# sys.path.append('/workspace/classification/code/')  # zjl
# sys.path.append('/nfs3-p1/hjc/classification/code/')  # vipa
import os
import shutil
from configs import config


# 遍历文件夹
def walk_file(path):
    count = 0
    for root, dirs, files in os.walk(path):
        print(root)

        for f in files:
            count += 1
            # print(os.path.join(root, f))

        for d in dirs:
            print(os.path.join(root, d))
    print("文件数量一共为:", count)


def count_images(path):
    for root, dirs, files in os.walk(path):
        print(root, len(files))


def copy_file(src, dst):
    path, name = os.path.split(dst)
    if not os.path.exists(path):
        os.makedirs(path)
    shutil.copyfile(src, dst)


if __name__ == '__main__':
    # walk_file(config.coco_images_2)
    count_images(config.data_mini_imagenet)
