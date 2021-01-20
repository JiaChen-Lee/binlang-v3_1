# Created by Jiachen Li at 2021/1/8 22:16
import os
import torch
import torch.utils.data as data
from data.transform import create_transform
from data.myDataset import MyDataset


def create_dataloader(root, super_cls, resized_size, batch_size, load_img_with="OpenCV", mode="val"):
    transform = create_transform(load_img_with=load_img_with, resized_size=resized_size)

    dataset = MyDataset(root=os.path.join(root, super_cls, mode),
                        transform=transform,
                        load_img_with=load_img_with)

    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=2)
    return dataloader


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
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=init_learning_rate,
                                    momentum=0.9,
                                    weight_decay=weight_decay)
    else:
        raise Exception("Error optimizer name or this {} optimizer is not supported".format(optimizer_name))

    return optimizer


def create_lr_scheduler(lr_scheduler_name, optimizer, T_max=5):
    if lr_scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=T_max,
                                                               eta_min=0,
                                                               last_epoch=-1)
    elif lr_scheduler_name == "ReduceLROnPlateau":
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
    else:
        raise Exception("Error scheduler name or this {} scheduler is not supported".format(lr_scheduler_name))

    return scheduler
