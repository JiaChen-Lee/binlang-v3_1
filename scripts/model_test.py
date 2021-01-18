# Created by Jiachen Li at 2020/12/29 22:13
import torch
# from torchvision import transforms
from torchvision.transforms import *
import os
from cvtorchvision import cvtransforms
from PIL import Image
from config import cfg
import numpy as np
import cv2

# def foo(path):
#     src = cv2.imread(path)
#     h, w, c = src.shape
#     img = np.zeros((1024, 1024, 3), np.uint8)
#     img[:h, 322:1024 - 322, :] = src
#     img = cv2.resize(img, (224, 224))
#     cv2.imwrite("/home/lijiachen/Projects/binlang-v3/tt.bmp", img)
#     # img = Image.open(path).convert('RGB')



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
transform = cvtransforms.Compose([
    cvtransforms.CenterCrop(1024),
    cvtransforms.Resize(size=(224, 224)),
    # cvtransforms.ToTensor(),
    # cvtransforms.Normalize([0.485, 0.456, 0.406],
    #                        [0.229, 0.224, 0.225]),
])


transforms_model = torch.nn.Sequential(
    CenterCrop(1024),
    transforms.Resize((224, 224)),
    # ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
)


# scripted_transforms = torch.jit.script(transforms)


def load_img(path):
    img = Image.open(path).convert('RGB')
    if cfg.transform is not None:
        img = cfg.transform(img)

    # img = cv2.imread(path,0)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img = transform(img)

    # img = torch.unsqueeze(img, dim=0)
    return img


if __name__ == '__main__':
    model = torch.load(
        "/home/lijiachen/Projects/binlang-v2/logs/2020-12-31_230509_resnest50/resnest50_epoch_130-acc_0.9215.pt")

    model.cuda()

    model.eval()
    root = "/home/lijiachen/data/binlang-v2/dataset-v7-rename_*_5/cut/val/"
    cls_name_list = os.listdir(root)
    cls_name_list.sort()
    cls_idx_map = {cls_name: idx for idx, cls_name in enumerate(cls_name_list)}
    for cls_name in os.listdir(root):
        cls_path = os.path.join(root, cls_name)
        num = 0
        for idx, img_name in enumerate(os.listdir(cls_path)):
            old_img_path = os.path.join(cls_path, img_name)
            # img = load_img(old_img_path)
            foo(old_img_path)
            # cv2.imwrite("/home/lijiachen/Projects/binlang-v3/cv_1.bmp",img)
            # img.save("/home/lijiachen/Projects/binlang-v3/pil_1.bmp", "bmp")
        #     img = img.cuda()
        #     out = model(img)
        #     m = torch.nn.Softmax(dim=1)
        #     out = m(out)
        #     pred = torch.max(out, 1)
        #
        #     pred_prob = pred[0].cpu().data.numpy()[0]
        #     pred_label = pred[1].cpu().data.numpy()[0]
        #     gt_label = cls_idx_map[cls_name]
        #     if gt_label == pred_label:
        #         num += 1
        #     # print("{} {} {} {:.4f}".format(gt_label, img_name, pred_label, pred_prob))
        # print("{} {:.4f}".format(cls_name, num / len(os.listdir(cls_path))))
