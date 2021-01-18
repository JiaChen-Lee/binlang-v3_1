"""
Tips:
    if CUDA out of memory. You can do
        @1 Turn down batch size
        @2 Switch to a smaller model
        @3 Resize input image to a smaller size
    
    If all the indicators have not been improved at the beginning of training. You can do
        @1 Lower learning rate
"""

hyperparameter_defaults = dict(
    num_classes=9,
    model="resnest50",
    pretrained=True,
    init_learning_rate=0.0001,
    batch_size=128,
    num_epochs=150,
    dataAug="mixup",
    optimizer="Adam",
    multi_gpu=True,
    weight_decay=0,
    scheduler_name="ReduceLROnPlateau",
    T_max=5,
    dataset_root="/home/lijiachen/data/binlang-v2/dataset-v7-rename_*_5/cut/",
    save_model=False,
    load_img_with="OpenCV",
    warmup_epoch=5,
    resized_size=224
)
