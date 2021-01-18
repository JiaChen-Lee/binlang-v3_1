# Created by Jiachen Li at 2020/12/26 22:50
# [:450] [-450:]
import cv2
import numpy as np
import os


def foo(path, dst_path):
    img = cv2.imread(path, 0)
    # print(img.shape)

    mask_1 = np.zeros(img.shape[0:2], dtype="uint8")
    cv2.rectangle(mask_1, (450, 0), (830, 1024), 255, -1)
    res = cv2.bitwise_and(img, img, mask=mask_1)
    # print(res.shape)

    mask_2 = 255 - mask_1
    res_2 = cv2.bitwise_or(res, mask_2)
    # cv2.imshow("img", res_2)
    # cv2.waitKey(0)
    cv2.imwrite(dst_path, res_2)


if __name__ == '__main__':
    root = "/home/lijiachen/data/binlang-v2/dataset-v1/"
    dst_root = "/home/lijiachen/data/binlang-v2/dataset-v2/"
    for cut_or_con_name in os.listdir(root):
        cut_or_con_path = os.path.join(root, cut_or_con_name)

        for cls_name in os.listdir(cut_or_con_path):
            cls_path = os.path.join(cut_or_con_path, cls_name)

            for channel_name in os.listdir(cls_path):
                channel_path = os.path.join(cls_path, channel_name)

                for img_name in os.listdir(channel_path):
                    img_path = os.path.join(channel_path, img_name)

                    dst_dir = os.path.join(dst_root,
                                           cut_or_con_name,
                                           cls_name,
                                           channel_name)
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    dst_path = os.path.join(dst_dir,
                                            img_name)
                    foo(img_path, dst_path)
