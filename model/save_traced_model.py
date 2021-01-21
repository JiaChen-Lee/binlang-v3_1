import torch
import torch.nn as nn
import cv2
from PIL import Image
from efficientnet_pytorch import EfficientNet
from config.cfg import hyperparameter_defaults as cfg
from data.transform import create_transform
from utils.dotdict import DotDict


cfg = DotDict(cfg)


def load_img(img_path, load_img_with):
    if load_img_with == "PIL":
        img = Image.open(img_path).convert('RGB')
    elif load_img_with == "OpenCV":
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise Exception("You must load image with OpenCV or PIL, but you passed {}".format(load_img_with))
    return img


def save_model(root, path, model_name, saved_model_name):
    img = load_img(img_path=path, load_img_with=cfg.load_img_with)
    transform = create_transform(load_img_with=cfg.load_img_with, resized_size=cfg.resized_size)
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.cuda()

    model_path = root + model_name
    model = torch.load(model_path)
    model.set_swish(False)
    model.eval()
    model.cuda()

    # model = EfficientNet.from_name('efficientnet-b3')
    # model.set_swish(memory_efficient=False)
    # num_ftrs = model._fc.in_features
    # model._fc = nn.Linear(num_ftrs, 9)
    # # model.load_state_dict(torch.load('Garbage/each_model/epoch_29.pth'))
    # #我是用四个GPU并行训练的，需要加这一句，如果是单GPU可以用上面的一句
    # model.load_state_dict(
    #     {k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
    traced_script_module = torch.jit.trace(model.module, img)
    # traced_script_module = torch.jit.script(model.module, img)
    traced_script_module.save(saved_model_name)


if __name__ == '__main__':
    if cfg.super_cls == "cut":
        img_path = "../data/dataset/cut/val/cut_50/channel_6_41.bmp"
        root = "/home/lijiachen/Projects/binlang-v3_1/logs/2021-01-18_194450_efficientnet-b3/"
        model_name = "efficientnet-b3_epoch_105-acc_0.9563.pt"
    elif cfg.super_cls == "con":
        img_path = "../data/dataset/cut/val/con_50/channel_3_48.bmp"
        root = "../logs/2021-01-18_212828_efficientnet-b3/"
        model_name = "efficientnet-b3_epoch_105-acc_0.9344.pt"
    else:
        raise Exception("Error super class name!")
    saved_model_name = "../traced_{}_{}.pt".format(cfg.super_cls, cfg.model)
    
    save_model(root=root, path=img_path, model_name=model_name, saved_model_name=saved_model_name)

    # path = "/home/lijiachen/data/binlang-v2/dataset-v7-rename_*_5/con/val/con_50/channel_3_48.bmp"
    # img = load_img(path)
    # img = img.cuda()
    # con_model_path = root + "2020-12-31_164033_resnest50/resnest50_epoch_180-acc_0.9064.pt"
    # model = torch.load(con_model_path)
    # traced_script_module = torch.jit.trace(model.module, img)
    # # parallel_model = nn.DataParallel(model)
    # # traced_script_module = torch.jit.trace(parallel_model.module, img)
    # traced_script_module.save("/home/lijiachen/Projects/binlang-v2/traced_resnest50_model_con.pt")
