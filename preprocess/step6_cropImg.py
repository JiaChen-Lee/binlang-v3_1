# Created by Jiachen Li at 2020/12/29 15:24
import cv2
import os


def foo(img_path, dst_img_path, crop_num):
    img = cv2.imread(img_path, 0)
    h, w = img.shape
    crop_img = img[:h, crop_num:w - crop_num]
    # crop_img = img[crop_num:w - crop_num, :h]
    cv2.imwrite(dst_img_path, crop_img)  # 传入错误的路径，居然不报错？！
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    # path = "test_data/1.bmp"
    # dst_path = "test/1.bmp"
    # foo(path,dst_path,450)
    crop_num = 450
    root = "/home/lijiachen/data/binlang-v2/dataset-v5-splitTrainVal/"
    dst_root = "/home/lijiachen/data/binlang-v2/dataset-v6-cropImg/"

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
                    dst_img_name = os.path.join(dst_img_path, img_name)
                    foo(img_path, dst_img_name, crop_num)
