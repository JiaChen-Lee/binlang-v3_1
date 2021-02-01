# Created by Jiachen Li at 2021/1/30 20:43
import cv2
import os


def foo(img_path, dst_img_path):
    img = cv2.imread(img_path)
    h, w, c = img.shape
    crop_img = img[:h, 128:w - 128]
    cv2.imwrite(dst_img_path, crop_img)


if __name__ == '__main__':
    root = "/home/lijiachen/data/binlang-v2/dataset-v9-convert2jpgFromv5/"
    dst_root = "/home/lijiachen/data/binlang-v2/dataset-v10-crop1024x1024/"

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