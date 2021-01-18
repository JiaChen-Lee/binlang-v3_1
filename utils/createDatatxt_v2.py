import random
import os
import shutil

label_dict = {
    "0_0": 0,
    "0_1": 1,
    "15_0": 2,
    "15_1": 3,
    "20_0": 4,
    "20_1": 5,
    "30_0": 6,
    "30_1": 7,
}
label_dict_0 = {
    "0_0": 0,
    "15_0": 1,
    "20_0": 2,
    "30_0": 3,
}
label_dict_1 = {
    "0_1": 0,
    "15_1": 1,
    "20_1": 2,
    "30_1": 3,
}

flag = ["0", "1", "all"]
path = "/home/lijiachen/data/binlang/preprocessed/"
Idx = 1
train_file = open(path + "train_{}.txt".format(flag[Idx]), "w")
test_file = open(path + "test_{}.txt".format(flag[Idx]), "w")
for cls_name in os.listdir(path):
    print(cls_name)
    if cls_name.endswith("txt"):
        continue
    if cls_name.endswith(flag[Idx]):  # 生成背面四类的txt，改成"0"则生成切面四类的txt
        size = len(os.listdir(path + cls_name))
        L = random.sample(os.listdir(path + cls_name), len(os.listdir(path + cls_name)))
        # print(L)
        train_size = int(size * 0.75)
        for i, img_name in enumerate(L):
            if i < train_size:
                train_file.write(path + "/" + cls_name + "/" + img_name + " " + str(label_dict_1[cls_name]))
                train_file.write("\n")
            else:
                test_file.write(path + "/" + cls_name + "/" + img_name + " " + str(label_dict_1[cls_name]))
                test_file.write("\n")
train_file.close()
test_file.close()
