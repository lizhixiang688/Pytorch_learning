# 名字：李志翔
# 创建时间:2021/5/31 11:11
#softmax的求导
import torch
a=torch.rand(3)
a.requires_grad_()
p=torch.softmax(a,dim=0)
print(torch.autograd.grad(p[1],[a],retain_graph=True))
print(torch.autograd.grad(p[2],[a],retain_graph=True))