import numpy as np
import math
from typing import List
import os
import argparse
import glob
import shutil


def list_files(path):
    files = os.listdir(path)
    return np.asarray(files)


def split_files(oldpath, newpath, classes):
    for name in classes:
        full_dir = os.path.join(os.getcwd(), f"{oldpath}/{name}")

        files = list_files(full_dir)
        total_file = np.size(files, 0)
        # We split data set into 3: train, validation and test

        train_size = math.ceil(total_file * 3 / 4)  # 75% for training

        validation_size = train_size + math.ceil(total_file * 1 / 8)  # 12.5% for validation
        test_size = validation_size + math.ceil(total_file * 1 / 8)  # 12.5x% for testing

        train = files[0:train_size]
        validation = files[train_size:validation_size]
        test = files[validation_size:]

        move_files(train, full_dir, f"train/{name}")
        move_files(validation, full_dir, f"validation/{name}")
        move_files(test, full_dir, f"test/{name}")


def move_files(files, old_dir, new_dir):
    new_dir = os.path.join(os.getcwd(), new_dir);
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for file in np.nditer(files):
        old_file_path = os.path.join(os.getcwd(), f"{old_dir}/{file}")
        new_file_path = os.path.join(os.getcwd(), f"{new_dir}/{file}")

        shutil.move(old_file_path, new_file_path)


if __name__ == '__main__':
    path = "/home/lijiachen/data/binlang/"
    list_files(path)
    classes = ["0_0",]
