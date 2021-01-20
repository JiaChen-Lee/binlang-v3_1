# Created by Jiachen Li at 2020/12/24 19:23
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2


class MyDataset(Dataset):
    def __init__(self, root, transform=None, load_img_with="PIL"):
        super(MyDataset, self).__init__()
        self.samples = []

        for cls_name in os.listdir(root):
            cls_path = os.path.join(root, cls_name)

            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)

                self.samples.append((img_path, cls_name))

        cls_name_list = list(os.listdir(root))
        cls_name_list.sort()
        self.cls_map_idx = {label: idx for idx, label in enumerate(cls_name_list)}

        self.transform = transform
        self.load_img_with = load_img_with

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        if self.load_img_with == "OpenCV":
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label_id = self.cls_map_idx[label]

        return img, label_id  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):
        return len(self.samples)

    def public_method(self, index):
        return self.__getitem__(index)

    def __load_img__(self, img_path):
        if self.load_img_with == "OpenCV":
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = Image.open(img_path).convert('RGB')
        return img


if __name__ == '__main__':
    root = cfg.dataset["train"]
    myData = MyDataset(root=root, transform=cfg.transform)
    img, label = myData.public_method(0)
