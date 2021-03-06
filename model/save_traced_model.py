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


def save_model(model_path, saved_model_name):
    img = torch.randn(1, 3, 224, 224)
    img = img.cuda()

    model = torch.load(model_path)
    model.eval()
    model.cuda()
    # model.module.set_swish(memory_efficient=False)

    traced_script_module = torch.jit.trace(model.module, img)
    traced_script_module.save(saved_model_name)

    # model.module.set_swish(memory_efficient=False)
    # torch.onnx.export(model.module, dummy_input, "/home/lijiachen/Projects/binlang-v3_1/test-b0.onnx", verbose=True)


if __name__ == '__main__':
    if cfg.super_cls == "cut":
        root = "/home/lijiachen/Projects/binlang-v3_1/logs/2021-02-04_002824_mobilenet_v2/"
        model_name = "cut_mobilenet_v2_epoch_485-acc_0.8861.pt"
    elif cfg.super_cls == "con":
        root = "/home/lijiachen/Projects/binlang-v3_1/logs/2021-02-03_080317_mobilenet_v2/"
        model_name = "con_mobilenet_v2_epoch_490-acc_0.8512.pt"
    else:
        raise Exception("Error super class name!")
    saved_model_name = "../traced_{}_{}.pt".format(cfg.super_cls, cfg.model)
    model_path = root + model_name
    save_model(model_path=model_path, saved_model_name=saved_model_name)

    # import timm
    #
    # model = timm.create_model(model_name="efficientnet_b0", pretrained=cfg.pretrained, num_classes=cfg.num_classes)
    # model.eval()
    # # model.cuda()
    # # model.module.set_swish(False)
    # img = torch.rand([1,3,224,224])
    # traced_script_module = torch.jit.trace(model, img)
    # # traced_script_module.save(saved_model_name)
    import geffnet

    # model = geffnet.create_model(model_name="mobilenetv3_large_100",
    #                              pretrained=True,
    #                              num_classes=9)
    # model.eval()
    # model.cuda()
    # # model.module.set_swish(False)
    # img = torch.rand([1,3,224,224])
    # img = img.cuda()
    # traced_script_module = torch.jit.trace(model, img)
    # # traced_script_module.save(saved_model_name)
