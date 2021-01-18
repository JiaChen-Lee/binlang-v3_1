# Created by Jiachen Li at 2021/1/14 22:33
import torch
import torch.utils.model_zoo as model_zoo
from model.registry import model_entrypoint
from efficientnet_pytorch import EfficientNet
import timm


def create_model(model_name,
                 pretrained,
                 num_classes):
    model_args = dict(pretrained=pretrained)
    if model_name.startswith("efficientnet"):
        model = EfficientNet.from_pretrained(model_name=model_name,
                                             num_classes=num_classes)
    elif model_name.startswith("vit"):
        model = timm.create_model(model_name=model_name,
                                  pretrained=pretrained,
                                  num_classes=num_classes)
    else:
        create_fn = model_entrypoint(model_name)
        model = create_fn(**model_args)
        feature = model.fc.in_features
        model.fc = torch.nn.Linear(feature, num_classes, bias=True)
    return model
