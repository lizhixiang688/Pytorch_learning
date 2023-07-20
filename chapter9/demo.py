# 名字：李志翔
# 创建时间:2021/8/3 16:31
# 基础的卷积神经网络
import torch
import torch.nn as nn

layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)
x = torch.rand(1, 1, 28, 28)
out = layer(x)

print(x.shape)
print(layer.weight.shape, layer.bias.shape)
