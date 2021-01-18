import torch
import torch.nn as nn
from PIL import Image
from config import cfg


def load_img(path):
    img = Image.open(path).convert('RGB')
    if cfg.transform is not None:
        img = cfg.transform(img)
    img = torch.unsqueeze(img, dim=0)
    return img


if __name__ == '__main__':
    root = "/home/lijiachen/Projects/binlang-v2/logs/"

    path = "/home/lijiachen/data/binlang-v2/dataset-v7-rename_*_5/cut/val/cut_50/channel_6_41.bmp"
    img = load_img(path)
    img = img.cuda()
    cut_model_path = root + "2020-12-31_230509_resnest50/resnest50_epoch_130-acc_0.9215.pt"
    model = torch.load(cut_model_path)
    traced_script_module = torch.jit.trace(model.module, img)
    # parallel_model = nn.DataParallel(model)
    # traced_script_module = torch.jit.trace(parallel_model.module, img)
    traced_script_module.save("/home/lijiachen/Projects/binlang-v2/traced_resnest50_model_cut.pt")

    # path = "/home/lijiachen/data/binlang-v2/dataset-v7-rename_*_5/con/val/con_50/channel_3_48.bmp"
    # img = load_img(path)
    # img = img.cuda()
    # con_model_path = root + "2020-12-31_164033_resnest50/resnest50_epoch_180-acc_0.9064.pt"
    # model = torch.load(con_model_path)
    # traced_script_module = torch.jit.trace(model.module, img)
    # # parallel_model = nn.DataParallel(model)
    # # traced_script_module = torch.jit.trace(parallel_model.module, img)
    # traced_script_module.save("/home/lijiachen/Projects/binlang-v2/traced_resnest50_model_con.pt")
