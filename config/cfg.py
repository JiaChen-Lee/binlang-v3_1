hyperparameter_defaults = dict(
    dataset_root="/home/lijiachen/Projects/binlang-v3_1/data/dataset/",
    super_cls="cut",
    num_classes=9,
    model="efficientnet-b0",
    pretrained=True,
    init_learning_rate=0.0001561,
    weight_decay=0.002329,
    batch_size=16,
    resized_size=224,
    num_workers=2,
    pin_memory=False,
    num_epochs=200,
    dataAug="mixup",
    load_img_with="OpenCV",
    optimizer="AdamW",
    warmup_epoch=5,
    scheduler_name="ReduceLROnPlateau",
    T_max=5,
    multi_gpu=True,
    use_amp=True,
    save_model=True,
)
"""
Tips:
    When DistributedDataParallel and mixup are used together, mixup will cause A GPU to consume more memory,
    because mixup operation need to be performed on this GPU
"""