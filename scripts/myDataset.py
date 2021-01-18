from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from config import cfg


class MyDataset(Dataset):
    def __init__(self, datatxt, transform=None):
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片

        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform

        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

    def public_method(self, index):
        return self.__getitem__(index)


if __name__ == '__main__':
    root = "/home/lijiachen/data/binlang/preprocessed/"
    myData = MyDataset(datatxt=root + 'train_0.txt', transform=cfg.transform_test)
    img, label = myData.public_method(0)
    # img.show()
    img.save("test.bmp")