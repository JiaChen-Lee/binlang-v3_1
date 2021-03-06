hyperparameter_defaults = dict(
    dataset_root="/home/lijiachen/data/binlang-v2/dataset-v11-resize224x224FromV7/",
    super_cls="cut",
    num_classes=9,
    model="mobilenet_v3_large",
    pretrained=True,
    init_learning_rate=0.0001387,
    weight_decay=0.00881,
    batch_size=32,
    resized_size=224,
    num_workers=2,  # 实测并不是像那个教程中说的设置成GPU数量的4倍的时候最快，反而会变慢，最佳就是2
    pin_memory=True,  # 实测True会稍快一点点
    num_epochs=700,
    dataAug="mixup",
    load_img_with="OpenCV",
    optimizer="AdamW",
    warmup_epoch=5,
    scheduler_name="CosineAnnealingLR",
    T_max=5,
    multi_gpu=True,
    gpu_ids="0",
    use_amp=False,  # 实测True的时候，显存占用并没有减少，反而运行时间边长
    save_model=True,
    save_interval=5
)
"""
Tips:
    When DistributedDataParallel and mixup are used together, mixup will cause A GPU to consume more memory,
    because mixup operation need to be performed on this GPU
"""