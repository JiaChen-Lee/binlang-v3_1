# Created by Jiachen Li at 2021/1/30 19:13
import cv2
import os


def foo(img_path, dst_img_path):
    img = cv2.imread(img_path)
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(dst_img_path, dst)


if __name__ == '__main__':
    root = "/home/lijiachen/data/binlang-v2/dataset-v5-splitTrainVal/"
    dst_root = "/home/lijiachen/data/binlang-v2/dataset-v9-convert2jpgFromv5/"

    for cls_name in os.listdir(root):
        cls_path = os.path.join(root, cls_name)

        for dataset_name in os.listdir(cls_path):
            dataset_path = os.path.join(cls_path, dataset_name)

            for sub_cls_name in os.listdir(dataset_path):
                sub_cls_path = os.path.join(dataset_path, sub_cls_name)

                for img_name in os.listdir(sub_cls_path):
                    img_path = os.path.join(sub_cls_path, img_name)
                    dst_img_path = os.path.join(dst_root,
                                                cls_name,
                                                dataset_name,
                                                sub_cls_name)
                    if not os.path.exists(dst_img_path):
                        os.makedirs(dst_img_path)
                    jpg_name = img_name.split(".")[0] + ".jpg"
                    dst_img_name = os.path.join(dst_img_path, jpg_name)
                    # print(dst_img_name)
                    foo(img_path, dst_img_name)
