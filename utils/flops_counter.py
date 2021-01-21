# Created by Jiachen Li at 2021/1/21 15:31
import os
import torchvision.models as models
import torch
from model import create_model
from ptflops import get_model_complexity_info
from config.cfg import hyperparameter_defaults
from utils.dotdict import DotDict

cfg = DotDict(hyperparameter_defaults)
with torch.cuda.device(0):
    # net = models.resnet18()
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    model = create_model(cfg.model, cfg.pretrained, cfg.num_classes)
    macs, params = get_model_complexity_info(model, (3, cfg.resized_size, cfg.resized_size), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
