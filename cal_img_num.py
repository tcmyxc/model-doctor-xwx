import os

def list_files_of_dir(dir_path):
    # with 语句可以偷懒，不用手动close
    cnt = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            cnt += 1
        
    print("==> img nums is ", cnt)

if __name__ == "__main__":
    dir_path = "/home/xwx/model-doctor-xwx/data/mini_imagenet/images/train/n01532829"

    list_files_of_dir(dir_path)