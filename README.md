# This project was migrated from the V3 release
## Changes:
In this version, `wandb-sweep` is introduced to carry out hyperparameter search
## Tips:
    if CUDA out of memory. You can do
        - Turn down batch size
        - Switch to a smaller model
        - Resize input image to a smaller size
    
    If all the indicators have not been improved at the beginning of training. You can do
        - Lower learning rate
## Optimal Hyperparameter and Performance
| dataset | model  |  input_size |   learning_rate | weight_decay | batch_size |val_acc|
|--------------|:--------:|:------:|:------:|:------:|:------:|:------:|
|cut  | efficientnet-b3 |300x300 |1.572e-4 | 1.165e-3| 64 | 96.46 |
|cut       | resnest50 |224x224 |1.14e-4 | 6.028e-3 | 128 | 92.27 |
|con  | efficientnet-b3|300x300|8.515e-5 | 9.883e-4 | 64 | 93.99 |
|con       | resnest50|224x224 |5.825e-5 | 1.134e-4 | 128 | 91.8|