# Created by Jiachen Li at 2021/1/4 22:39
import torch

x = torch.tensor([1, 2, 3, 4])
print(x.shape)
y1 = torch.unsqueeze(x, 0)
print(y1.shape)
y2 = torch.unsqueeze(x, 1)
print(y2.shape)
