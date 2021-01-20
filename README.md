# Binlang Classification

## What's New
This project was migrated from the V3 release.<br/>
In this version, `wandb-sweep` is introduced to carry out hyperparameter search

## Introduction
This repository is used to develop visual recognition algorithm for binlang classification project

## Getting Started
You can modify hyperparameter in [cfg.py](config/cfg.py)

## Models
* EfficientNet - https://arxiv.org/abs/1905.11946
* ResNeSt - https://arxiv.org/abs/2004.08955
* CBAM - https://arxiv.org/abs/1807.06521
* SKNet - https://arxiv.org/abs/1903.06586
* ViT - https://arxiv.org/abs/2010.11929
* ResNet - https://arxiv.org/abs/1512.03385

## DataAug
* Mixup - https://arxiv.org/abs/1710.09412
* CutMix - https://arxiv.org/abs/1905.04899

## Optimizer
* Adam - https://arxiv.org/abs/1412.6980
* AdamW - https://arxiv.org/abs/1711.05101

## Train, Validation, Save Model
- You can train with [main.py](main.py).
- You can validate with [inference.py](scripts/inference.py).
- You can use [save_traced_model.py](model/save_traced_model.py) to save the model that used in LibTorch.

## Hyperparameter Optimization
- If you want to find optimal hyperparameter, you can use [sweep.yaml](config/sweep.yaml).
- From [here](https://docs.wandb.ai/sweeps), you can get details for sweep.

## Tips
    if CUDA out of memory. You can do
        - Turn down batch size
        - Switch to a smaller model
        - Resize input image to a smaller size
    
    If all the indicators have not been improved at the beginning of training. You can do
        - Lower learning rate

## Results
| dataset | model  |  input_size |   optimizer |   scheduler |   learning_rate | weight_decay | batch_size |val_acc|
|--------------|:--------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|cut  | efficientnet-b3 |300x300 |AdamW |ReduceLROnPlateau |1.572e-4 | 1.165e-3| 64 | 96.46 |
|cut       | resnest50 |224x224 |AdamW |CosineAnnealingLR |1.14e-4 | 6.028e-3 | 128 | 92.27 |
|con  | efficientnet-b3|300x300|AdamW |ReduceLROnPlateau |8.515e-5 | 9.883e-4 | 64 | 93.99 |
|con       | resnest50|224x224 |AdamW |ReduceLROnPlateau|5.825e-5 | 1.134e-4 | 128 | 91.8|

## Dataset
About datasets detail, you can look [DATA.md](data/DATA.md)