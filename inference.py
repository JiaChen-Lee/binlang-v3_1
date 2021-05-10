# Created by Jiachen Li at 2020/12/29 22:13
import time
import torch
import os
import pprint
from config.cfg import hyperparameter_defaults as cfg
from utils.cls_map_idx import cls_map_idx
from scripts.create_component import create_dataloader
from train_old import val
from utils.dotdict import DotDict

cfg = DotDict(cfg)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def inference(path):
    model = torch.load(path)
    print(model)
    model.cuda()
    model.eval()
    dataloader = create_dataloader(root=cfg.dataset_root,
                                   super_cls=cfg.super_cls,
                                   resized_size=cfg.resized_size,
                                   batch_size=cfg.batch_size,
                                   num_workers=cfg.num_workers,
                                   pin_memory=cfg.pin_memory,
                                   load_img_with=cfg.load_img_with,
                                   mode="val")
    val_acc, val_cls_acc = val(model, dataloader, cfg.num_classes)
    cls_idx_map = cls_map_idx(cfg.dataset_root)
    ret = {}
    for i in range(cfg.num_classes):
        ret['{}_acc'.format(cls_idx_map[i])] = val_cls_acc[i]
    return ret


if __name__ == '__main__':
    # pprint.pprint(cfg)
    if cfg.super_cls == "cut":
        img_path = "../data/dataset/cut/val/cut_50/channel_6_41.bmp"
        # root = "/home/lijiachen/Projects/binlang-v3_1/logs/2021-01-24_230437_efficientnet-b0/"
        root = "/home/lijiachen/Projects/binlang-v3_1/logs/2021-05-10_211434_mobilenet_v2/"
        # model_name = "cut_efficientnet-b0_epoch_120-acc_0.9239.pt"
        model_name = "cut_mobilenet_v2_epoch_160-acc_0.9439.pt"
    elif cfg.super_cls == "con":
        img_path = "../data/dataset/cut/val/con_50/channel_3_48.bmp"
        # root = "/home/lijiachen/Projects/binlang-v3_1/logs/2021-01-18_212828_efficientnet-b3/"
        # model_name = "efficientnet-b3_epoch_105-acc_0.9344.pt"
        root = "/home/lijiachen/Projects/binlang-v3_1/logs/2021-05-10_171115_mobilenet_v2/"
        model_name = "con_mobilenet_v2_epoch_210-acc_0.8867.pt"
    else:
        raise Exception("{} is error super class name!".format(cfg.super_cls))

    start = time.time()
    result = inference(root + model_name)
    end = time.time()
    sum_time = end - start
    print(sum_time)

    pprint.pprint(result)
