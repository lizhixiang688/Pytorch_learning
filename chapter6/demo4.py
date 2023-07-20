# 名字：李志翔
# 创建时间:2021/7/31 17:07
#多层感知机
import torch
from torch.nn import functional as F

x=torch.randn(1,10)
w=torch.randn(2,10,requires_grad=True)
o=torch.sigmoid(x@w.t())
loss=F.mse_loss(torch.ones(1,2),o)
loss.backward()
print(w.grad)