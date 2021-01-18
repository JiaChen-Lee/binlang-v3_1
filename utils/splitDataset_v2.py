import random
import os
import shutil

path = "/home/lijiachen/data/binlang/"
train_target = "/home/lijiachen/data/binlangDataset/trainData/"
test_target = "/home/lijiachen/data/binlangDataset/testData/"
for cls in os.listdir(path):
    print(cls)
    size = len(os.listdir(path + cls))
    L = random.sample(os.listdir(path + cls), len(os.listdir(path + cls)))
    print(L)
    train_size = int(size * 0.75)
    for i, img_name in enumerate(L):
        if not os.path.exists(train_target + cls):
            os.makedirs(train_target + cls)
        if not os.path.exists(test_target + cls):
            os.makedirs(test_target + cls)
        if i < train_size:
            shutil.copy(path + cls + "/" + img_name, train_target + cls + "/" + img_name)
        else:
            shutil.copy(path + cls + "/" + img_name, test_target + cls + "/" + img_name)
