# Created by Jiachen Li at 2021/1/18 21:02
import os
from config.cfg import hyperparameter_defaults as cfg
from utils.dotdict import DotDict

cfg = DotDict(cfg)


def cls_map_idx(root):
    cls_name_list = os.listdir(os.path.join(root, cfg.super_cls, "train"))
    cls_name_list.sort()
    cls_idx_map = {idx: cls_name for idx, cls_name in enumerate(cls_name_list)}
    return cls_idx_map
