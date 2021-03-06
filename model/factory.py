# Created by Jiachen Li at 2021/1/14 22:33
import torch
import torch.nn as nn
from model.registry import model_entrypoint
from efficientnet_pytorch import EfficientNet
# import timm


def create_model(model_name,
                 pretrained,
                 num_classes):
    if model_name.startswith("efficientnet"):
        model = EfficientNet.from_pretrained(model_name=model_name,
                                             num_classes=num_classes)
        model.set_swish(memory_efficient=False)
    elif model_name.startswith("vit"):
        model = timm.create_model(model_name=model_name,
                                  pretrained=pretrained,
                                  num_classes=num_classes)
    elif model_name.startswith("sknet"):
        create_fn = model_entrypoint(model_name)
        model = create_fn(num_classes=num_classes)
    elif model_name.startswith("se_resnet"):
        create_fn = model_entrypoint(model_name)
        model = create_fn(num_classes=num_classes)
    elif model_name.startswith("mobilenet"):
        create_fn = model_entrypoint(model_name)
        model = create_fn(pretrained=pretrained)
        feature = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(feature, num_classes)
    else:
        create_fn = model_entrypoint(model_name)
        model = create_fn(pretrained=pretrained)
        feature = model.fc.in_features
        model.fc = torch.nn.Linear(feature, num_classes, bias=True)
    return model

