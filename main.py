import time
import os
import torch
from warmup_scheduler import GradualWarmupScheduler
import wandb

from model import create_model
from create_component import create_dataloader, create_optimizer, create_lr_scheduler
from train import train_epoch, val
from config.cfg import hyperparameter_defaults

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# # initialize wandb
wandb.init(config=hyperparameter_defaults)
cfg = wandb.config

cls_name_list = os.listdir(os.path.join(cfg.dataset_root, "train"))
cls_name_list.sort()
cls_idx_map = {idx: cls_name for idx, cls_name in enumerate(cls_name_list)}

# ================ logs ===================
cur_time = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
logs_path = "./logs/{}_{}".format(cur_time, cfg.model)
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

if __name__ == '__main__':
    # ======================= set model =======================
    model = create_model(cfg.model, cfg.pretrained, cfg.num_classes)

    if cfg.multi_gpu:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    model.cuda()
    wandb.watch(models=model)

    # ================= dataloader ====================
    train_dataloader, test_dataloader = create_dataloader(cfg.dataset_root,
                                                          cfg.resized_size,
                                                          cfg.batch_size)

    # ===================== optimize ========================
    optimizer = create_optimizer(cfg.optimizer,
                                 model,
                                 cfg.init_learning_rate,
                                 cfg.weight_decay)
    scheduler = create_lr_scheduler(cfg.scheduler_name,
                                    optimizer,
                                    cfg.T_max)
    scheduler_warmup = GradualWarmupScheduler(optimizer,
                                              multiplier=1.5,
                                              total_epoch=cfg.warmup_epoch,
                                              after_scheduler=scheduler)

    # ===================== loss function ====================
    loss_func = torch.nn.CrossEntropyLoss()

    # ===================== training ====================
    val_acc = 0
    for epoch in range(cfg.num_epochs):

        print('epoch {}'.format(epoch + 1))
        print(optimizer.state_dict()['param_groups'][0]['lr'])

        loss, acc = train_epoch(model, optimizer, train_dataloader, loss_func, cfg.dataAug)
        val_acc, val_cls_acc = val(model, test_dataloader, cfg.num_classes)

        scheduler_warmup.step(epoch, metrics=val_acc)

        if cfg.save_model and (epoch + 1) % 5 == 0:
            checkpoint_path = logs_path + "/{}_epoch_{}-acc_{:.4f}.pt".format(cfg.model, epoch + 1, val_acc)
            torch.save(model, checkpoint_path)

        # ==================== logs with wandb ======================
        metrics = {"Train/loss": loss,
                   "Train/acc": acc,
                   "Test/Acc_Top1": val_acc,
                   "lr": optimizer.state_dict()['param_groups'][0]['lr']}
        for i in range(cfg.num_classes):
            metrics['Test/{}_acc'.format(cls_idx_map[i])] = val_cls_acc[i]
        wandb.log(metrics)
