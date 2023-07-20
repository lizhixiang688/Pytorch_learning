# 名字：李志翔
# 创建时间:2021/8/1 11:35
#crossEntropy
import torch
from torch.nn import functional as F


x=torch.randn(1,784)
w=torch.randn(10,784)
logits=x@w.t()
pred=F.softmax(logits,dim=1)
pred_log=torch.log(pred)

print(F.cross_entropy(logits,torch.tensor([3])))
print(F.nll_loss(pred_log,torch.tensor([3])))


print(x)