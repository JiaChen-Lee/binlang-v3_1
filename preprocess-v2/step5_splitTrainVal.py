# Created by Jiachen Li at 2020/12/29 14:55
import os
import random
import shutil


def foo(root, dst_root, ratio):
    for cls_name in os.listdir(root):
        cls_path = os.path.join(root, cls_name)

        for sub_cls_name in os.listdir(cls_path):
            sub_cls_path = os.path.join(cls_path, sub_cls_name)

            img_name_list = os.listdir(sub_cls_path)
            random.shuffle(img_name_list)
            index = int(ratio / (ratio + 1) * len(img_name_list))
            train = img_name_list[:index]
            val = img_name_list[index:]

            for img_name in train:
                dst_img_path = os.path.join(dst_root, cls_name, "train", sub_cls_name)
                if not os.path.exists(dst_img_path):
                    os.makedirs(dst_img_path)
                img_path = os.path.join(root, cls_name, sub_cls_name, img_name)
                shutil.copy(img_path, dst_img_path)
            for img_name in val:
                dst_img_path = os.path.join(dst_root, cls_name, "val", sub_cls_name)
                if not os.path.exists(dst_img_path):
                    os.makedirs(dst_img_path)
                img_path = os.path.join(root, cls_name, sub_cls_name, img_name)
                shutil.copy(img_path, dst_img_path)


if __name__ == '__main__':
    root = "/home/lijiachen/data/binlang-v2/dataset-v4-mergeChannel/"
    dst_root = "/home/lijiachen/data/binlang-v2/dataset-v5-splitTrainVal/"
    ratio = 3
    foo(root, dst_root, ratio)
