hyperparameter_defaults = dict(
    dataset_root="/home/lijiachen/Projects/binlang-v3_1/data/dataset/",
    super_cls="con",
    num_classes=9,
    model="mobilenet_v3_large",
    pretrained=True,
    init_learning_rate=0.0001,
    weight_decay=0.0,
    batch_size=128,
    resized_size=224,
    num_workers=2,  # 实测并不是像那个教程中说的设置成GPU数量的4倍的时候最快，反而会变慢，最佳就是2
    pin_memory=True,  # 实测True会稍快一点点
    num_epochs=400,
    dataAug="mixup",
    load_img_with="OpenCV",
    optimizer="Adam",
    warmup_epoch=5,
    scheduler_name="CosineAnnealingLR",
    T_max=5,
    multi_gpu=True,
    use_amp=False,  # 实测True的时候，显存占用并没有减少，反而运行时间边长
    save_model=True,
)
"""
Tips:
    When DistributedDataParallel and mixup are used together, mixup will cause A GPU to consume more memory,
    because mixup operation need to be performed on this GPU
"""