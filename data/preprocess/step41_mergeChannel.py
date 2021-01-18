# Created by Jiachen Li at 2020/12/28 21:23
import os
import shutil


def foo(root, dst_root):
    for cls_name in os.listdir(root):
        cls_path = os.path.join(root, cls_name)

        if not cls_name.endswith("backup"):
            for sub_cls_name in os.listdir(cls_path):
                sub_cls_path = os.path.join(cls_path, sub_cls_name)

                if os.path.isdir(sub_cls_path) and sub_cls_name[-2] != "0":
                    for channel_name in os.listdir(sub_cls_path):
                        channel_path = os.path.join(sub_cls_path, channel_name)

                        for img_name in os.listdir(channel_path):
                            img_path = os.path.join(channel_path, img_name)

                            dst_img_path = os.path.join(dst_root, cls_name, sub_cls_name)
                            if not os.path.exists(dst_img_path):
                                os.makedirs(dst_img_path)
                            shutil.copy(img_path, dst_img_path)

                            dst_img_name = os.path.join(dst_img_path, img_name)
                            new_img_name = channel_name + '_' + img_name
                            new_img_path = os.path.join(dst_img_path, new_img_name)
                            os.rename(dst_img_name, new_img_path)


if __name__ == '__main__':
    root = "/home/lijiachen/data/binlang-v2/dataset-v3-addCls/"
    dst_root = "/home/lijiachen/data/binlang-v2/dataset-v4/"
    foo(root, dst_root)
