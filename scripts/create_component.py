# Created by Jiachen Li at 2021/1/8 22:16
import os
import torch
import torch.utils.data as data
# import timm
from torchvision import transforms
from cvtorchvision import cvtransforms

from data.myDataset import MyDataset

# from efficientnet_pytorch import EfficientNet
# from model.myResNeSt import resnest50
# from model.myResNeSt import resnest101
# from model.myCBAM.cbam import resnet18_cbam
# from model.myCBAM.cbam import resnet34_cbam
# from model.myCBAM.cbam import resnet50_cbam


# def create_model(model_name, pretrained, num_classes):
#     resnest_model = {"resnest50": resnest50,
#                      "resnest101": resnest101}
#     cbam_model = {"resnet18_cbam": resnet18_cbam,
#                   "resnet34_cbam": resnet34_cbam,
#                   "resnet50_cbam": resnet50_cbam}
#     resnet_model = {"resnet18": models.resnet18,
#                     "resnet34": models.resnet34,
#                     "resnet50": models.resnet50}
#     if model_name == "resnest50":
#         model = resnest50(pretrained=pretrained, num_classes=num_classes)  # resnest18无预训练模型
#         # feature = model.fc.in_features
#         # model.fc = torch.nn.Linear(feature, num_classes, bias=True)
#     elif model_name.startswith("efficientnet"):
#         model = EfficientNet.from_pretrained(model_name=model_name,
#                                              num_classes=num_classes)
#     elif model_name == "cbam":
#         model = resnet18_cbam(pretrained=pretrained, num_classes=num_classes)  # 学习速度很快！！！
#         # feature = model.fc.in_features
#         # model.fc = torch.nn.Linear(feature, num_classes, bias=True)
#     elif model_name.startswith("vit"):
#         model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
#     else:
#         model = models.resnet18(pretrained=pretrained, num_classes=num_classes)  # 作为baseline对比其他模型
#         # feature = model.fc.in_features
#         # model.fc = torch.nn.Linear(feature, num_classes, bias=True)
#
#     return model
#

def create_dataloader(root, resized_size, batch_size, load_img_with="PIL"):
    if load_img_with == "OpenCV":
        transform = cvtransforms.Compose([
            cvtransforms.CenterCrop(1024),
            cvtransforms.Resize((resized_size, resized_size)),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.RandomVerticalFlip(),
            cvtransforms.ToTensor(),
            cvtransforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(1024),
            transforms.Resize((resized_size, resized_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    train_dataset = MyDataset(root=os.path.join(root, "train"),
                              transform=transform,
                              load_img_with=load_img_with)
    test_dataset = MyDataset(root=os.path.join(root, "val"),
                             transform=transform,
                             load_img_with=load_img_with)

    train_dataloader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2)
    test_dataloader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=2)
    return train_dataloader, test_dataloader


def create_optimizer(optimizer_name, model, init_learning_rate, weight_decay):
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=init_learning_rate,
                                     weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(params=model.parameters(),
                                      lr=init_learning_rate,
                                      betas=(0.9, 0.999),
                                      eps=1e-08,
                                      weight_decay=weight_decay,
                                      amsgrad=False)
    else:
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=init_learning_rate,
                                    momentum=0.9,
                                    weight_decay=weight_decay)
    return optimizer


def create_lr_scheduler(lr_scheduler_name, optimizer, T_max=5):
    if lr_scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=T_max,
                                                               eta_min=0,
                                                               last_epoch=-1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='max',
                                                               factor=0.1,
                                                               patience=10,
                                                               verbose=False,
                                                               threshold=1e-4,
                                                               threshold_mode='rel',
                                                               cooldown=0,
                                                               min_lr=0,
                                                               eps=1e-08)
    return scheduler
