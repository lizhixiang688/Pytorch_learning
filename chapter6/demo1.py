# 名字：李志翔
# 创建时间:2021/7/30 16:38
#mseloss（均方差）的求导
import torch
from torch.nn import functional as F

x=torch.ones(1)
w=2*torch.ones(1)
w.requires_grad_()
mse=F.mse_loss(torch.ones(1),x*w)
mse.backward()
print(w.grad)