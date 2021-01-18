# Created by Jiachen Li at 2020/12/30 22:38
import torch

label = torch.tensor([[1, 2, 3]])
# a = label.view(3, 1)
# print(a)

pred = torch.tensor([[1, 2], [0, 0], [0, 0]])
pred = pred.t()
b = label.view(1, -1).expand_as(pred)
print(pred.eq(b).sum(-1).sum().item())

# def accuracy(output, target, topk=(1,)):
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1,-1).expand_as(pred))
#
#     res = []
#
print()
c = torch.zeros(10)
d = torch.ones(10)
print(c + d)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
