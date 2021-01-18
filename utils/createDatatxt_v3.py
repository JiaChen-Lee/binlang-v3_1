# Created by Jiachen Li at 2020/12/24 20:16
# 构建dataset_v2，基于preprocessed
import os
import shutil

root = "/home/lijiachen/data/binlang/dataset_v2/"


def foo(path):
    with open(path) as f:
        for line in f.readlines():
            img_path, label_idx = line.split()
            cls_name = img_path.split("/")[-2]
            dst_path = os.path.join(root, "cut", "test",cls_name)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            shutil.copy(img_path, dst_path)


if __name__ == '__main__':
    txt_path = "/home/lijiachen/data/binlang/preprocessed/cut/test_1.txt"
    foo(txt_path)
