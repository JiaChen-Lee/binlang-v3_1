import torch
import numpy as np
from data.dataAug.mixup import mixup_data
from data.dataAug.mixup import mixup_criterion
from data.dataAug.cutmix import cutmix_data
from data.dataAug.cutmix import cutmix_criterion


def train_epoch(model, optimizer, train_dataloader, loss_func, data_aug=None):
    # ================== initial =======================
    train_loss = 0.
    train_acc = 0.
    train_num = 0

    # ==================== batch =====================
    for data, label in train_dataloader:
        # ================= move data to GPU =================
        data = data.cuda()
        label = label.cuda()

        # =============== data augment =====================
        if data_aug == "mixup":
            data, label_a, label_b, lam = mixup_data(data, label)
            out = model(data)
            loss = mixup_criterion(loss_func, out, label_a, label_b, lam)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = lam * (pred == label_a).sum() + (1 - lam) * (pred == label_b).sum()
            train_acc += train_correct.item()
        elif data_aug == "cutmix":
            data, label_a, label_b, lam = cutmix_data(data, label)
            out = model(data)
            loss = cutmix_criterion(loss_func, out, label_a, label_b, lam)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = lam * (pred == label_a).sum() + (1 - lam) * (pred == label_b).sum()
            train_acc += train_correct.item()
        else:
            # ================== loss function without data augment ===================
            out = model(data)
            loss = loss_func(out, label)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]

            train_correct = (pred == label).sum()
            train_acc += train_correct.item()

        # ================= backward ====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_num += len(data)
    train_loss = train_loss / train_num
    train_acc = train_acc / train_num

    return train_loss, train_acc


def val(model, test_dataloader, num_classes):
    # ================== initial =======================
    model.eval()
    eval_acc = 0.
    test_num = 0
    correct = [0 for _ in range(num_classes)]
    total = [0 for _ in range(num_classes)]
    # ================== batch =======================
    for data, label in test_dataloader:
        data = data.cuda()
        label = label.cuda()
        # ==================== forward ======================
        out = model(data)
        pred = torch.max(out, 1)[1]

        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()

        # ================ every class acc =================
        res = pred == label
        for label_idx in range(len(label)):
            label_single = label[label_idx]
            correct[label_single] += res[label_idx].item()
            total[label_single] += 1

        test_num += len(data)
    val_acc = eval_acc / test_num
    val_cls_acc = np.array(correct) / np.array(total)

    return val_acc, val_cls_acc

