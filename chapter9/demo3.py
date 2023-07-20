# 名字：李志翔
# 创建时间:2021/8/4 10:32
#Batch_Norm

import torch.nn as nn
import torch

x=torch.rand(100,16,784)
layer=nn.BatchNorm1d(16)     #一维的
out=layer(x)
print(layer.running_mean)
print(layer.running_var)


x=torch.randn(1,16,7,7)
layer2=nn.BatchNorm2d(16)     #二维的
out2=layer2(x)
print(layer2.weight)
print(layer2.bias)




