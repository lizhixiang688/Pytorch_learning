# 名字：李志翔
# 创建时间:2021/8/4 10:08
#池化与采样
import torch
import torch.nn as nn
import torch.nn.functional as F

x=torch.randn(1,1,28,28)
layer1=nn.Conv2d(1,3,kernel_size=3,padding=0,stride=1)
out=layer1(x)
print(out.shape)
layer2=nn.MaxPool2d(2,stride=2)
out2=layer2(out)
layer3=nn.ReLU(inplace=True)
out3=layer3(out2)
out3=F.interpolate(out3,scale_factor=2,mode='nearest')
print(out3.shape)