# Created by Jiachen Li at 2021/1/22 16:38
import time
import os
import torch
from warmup_scheduler import GradualWarmupScheduler
import wandb

from model import create_model
from scripts.create_component import create_dataloader, create_optimizer, create_lr_scheduler
from train_old import train_epoch, val
from config.cfg import hyperparameter_defaults
from utils.cls_map_idx import cls_map_idx

# # initialize wandb
wandb.init(config=hyperparameter_defaults)
cfg = wandb.config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

cls_idx_map = cls_map_idx(cfg.dataset_root)

# ================ logs ===================
cur_time = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
logs_path = "./logs/{}_{}".format(cur_time, cfg.model)
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

if __name__ == '__main__':
    # ======================= set model =======================
    model = create_model(cfg.model, cfg.pretrained, cfg.num_classes)

    if cfg.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.cuda()
    wandb.watch(models=model)

    # ================= dataloader ====================
    train_dataloader = create_dataloader(root=cfg.dataset_root,
                                         super_cls=cfg.super_cls,
                                         resized_size=cfg.resized_size,
                                         batch_size=cfg.batch_size,
                                         num_workers=cfg.num_workers,
                                         pin_memory=cfg.pin_memory,
                                         load_img_with=cfg.load_img_with,
                                         mode="train")
    test_dataloader = create_dataloader(root=cfg.dataset_root,
                                        super_cls=cfg.super_cls,
                                        resized_size=cfg.resized_size,
                                        batch_size=cfg.batch_size,
                                        num_workers=cfg.num_workers,
                                        pin_memory=cfg.pin_memory,
                                        load_img_with=cfg.load_img_with,
                                        mode="val")

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

        print("Epoch {}/{}".format(epoch+1, cfg.num_epochs))
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print("lr: {:.6f}".format(lr))
        import time

        start = time.time()
        loss, acc = train_epoch(model, optimizer, train_dataloader, loss_func,
                                data_aug=cfg.dataAug, use_amp=cfg.use_amp)
        val_acc, val_cls_acc = val(model, test_dataloader, cfg.num_classes)
        end = time.time()

        sum_time = end - start
        print("{:.2f}s/epoch".format(round(sum_time, 2)))

        # scheduler_warmup.step(epoch, metrics=val_acc)
        scheduler_warmup.step(metrics=val_acc)

        if cfg.save_model and (epoch + 1) % cfg.save_interval == 0:
            checkpoint_name = "/{}_{}_epoch_{}-acc_{:.4f}.pt".format(cfg.super_cls, cfg.model, epoch + 1, val_acc)
            checkpoint_path = logs_path + checkpoint_name
            torch.save(model, checkpoint_path)
            # print(wandb.run.dir)
            # wandb.save(os.path.join(wandb.run.dir, checkpoint_name))

        # ==================== logs with wandb ======================
        metrics = {"Train/loss": loss,
                   "Train/acc": acc,
                   "Test/Acc_Top1": val_acc,
                   "lr": optimizer.state_dict()['param_groups'][0]['lr']}
        for i in range(cfg.num_classes):
            metrics['Test/{}_acc'.format(cls_idx_map[i])] = val_cls_acc[i]
        wandb.log(metrics)
