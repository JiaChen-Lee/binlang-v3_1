# Binlang Classification

## What's New
### Feb 01, 2021
- For lower latency, introduce mobilenet series model
### Dec 30, 2020
- Use `wandb` instead of `tensorboard` to log.
  `wandb-sweep` is introduced to implement hyperparameter search

## Introduction
This repository is used to develop visual recognition algorithm for binlang classification project

## Getting Started
You can modify hyperparameter in [cfg.py](config/cfg.py)

## Models
* MobileNet:
  * v3 - https://arxiv.org/abs/1905.02244
  * v2 - https://arxiv.org/abs/1801.04381
  * v1 - https://arxiv.org/abs/1704.04861
* ViT - https://arxiv.org/abs/2010.11929
* EfficientNet - https://arxiv.org/abs/1905.11946
* ResNeSt - https://arxiv.org/abs/2004.08955
* CBAM - https://arxiv.org/abs/1807.06521
* SKNet - https://arxiv.org/abs/1903.06586
* SENet - https://arxiv.org/abs/1709.01507
* ResNet - https://arxiv.org/abs/1512.03385

## Dataset
About datasets detail, you can look [DATA.md](data/DATA.md)

## DataAug
* Mixup - https://arxiv.org/abs/1710.09412
* CutMix - https://arxiv.org/abs/1905.04899

## Optimizer
* Adam - https://arxiv.org/abs/1412.6980
* AdamW - https://arxiv.org/abs/1711.05101

## Train, Validation, Save Model
- You can train with [main.py](main.py).
- You can validate with [inference.py](inference.py).
- You can use [save_traced_model.py](model/save_traced_model.py) to save the model that used in LibTorch.

## Hyperparameter Optimization
- If you want to find optimal hyperparameter, you can use [sweep.yaml](config/sweep.yaml).
- From [here](https://docs.wandb.ai/sweeps), you can get how to sweep.

## Tips
    If CUDA out of memory. You can do
        - Turn down batch size
        - Switch to a smaller model
        - Resize input image to a smaller size
    
    If all the indicators have not been improved at the beginning of training. You can do
        - Lower learning rate

    Small batch sizes generally converge faster and learn better than large batch sizes?
## Notes
- To be consistent with the transform implemented in C++, [cvtransforms](https://pypi.org/project/opencv-torchvision-transforms-yuzhiyang/) is recommended.

## Results on Cut
| Network          | MAdds | Params|Pretrained|Input Size|Optimizer|  Schedule       |Batch Size|Epoch|Top-1|Latency|
|------------------|:-----:|:-----:|:--------:|:--------:|:-------:|:---------------:|:--------:|:---:|:---:|:-----:|
|mobilenet_v3_large| 0.23 G| 4.21 M|    True  |   240x240|AdamW    |CosineAnnealingLR| 128      | 700 |94.51|None   |
|mobilenet_v2      | 0.32 G| 2.24 M|    True  |   240x240|AdamW    |CosineAnnealingLR|  32      | 700 |91.21|110ms  |
|efficientnet-b3   | 0.05 G|10.71 M|    True  |   300x300|AdamW    |ReduceLROnPlateau| 128      | 150 |96.46|None   |
|efficientnet-b0   | 0.01 G| 4.02 M|    True  |   240x240|AdamW    |ReduceLROnPlateau| 128      | 150 |93.75|700ms  |
|resnest50         | 5.41 G|25.45 M|    True  |   240x240|AdamW    |CosineAnnealingLR| 128      |200  |92.27|750ms  |
|resnet50          | 4.12 G|23.53 M|    True  |   240x240|AdamW    |CosineAnnealingLR| 128      | 200 |85.13|None   |
## Results on Con
| Network          | MAdds | Params|Pretrained|Input Size|Optimizer|  Schedule       |Batch Size|Epoch|Top-1|Latency|
|------------------|:-----:|:-----:|:--------:|:--------:|:-------:|:---------------:|:--------:|:---:|:---:|:-----:|
|mobilenet_v3_large| 0.23 G| 4.21 M|    True  |   224x224|AdamW    |CosineAnnealingLR| 128      |700  |91.31|None   |
|mobilenet_v2      | 0.32 G| 2.24 M|    True  |   224x224|AdamW    |CosineAnnealingLR| 32       |700  |86.94|110ms  |
|efficientnet-b3   | 0.05 G|10.71 M|    True  |   300x300|AdamW    |ReduceLROnPlateau| 128      |150  |93.99|None   |
|efficientnet-b0   | 0.01 G| 4.02 M|    True  |   240x240|AdamW    |ReduceLROnPlateau| 128      |150  |90.7 |700ms  |
|resnest50         | 5.41 G|25.45 M|    True  |   240x240|AdamW    |ReduceLROnPlateau| 128      |150  |91.8 |750ms  |
|resnet50          | 4.12 G|23.53 M|    True  |   240x240|AdamW    |CosineAnnealingLR| 128      |200  |80.19|None   |

### Notes: 
- All the above parameters are obtained by a sweep search, and None indicates that it has not been tested yet




