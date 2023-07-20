# 名字：李志翔
# 创建时间:2021/8/6 11:18
# 简单的RNN与RNNCell

import torch.nn as nn
import torch

rnn = nn.RNN(100, 20, num_layers=4)
print(rnn)

x = torch.randn(10, 3, 100)  # 10个单词，3个句子（batch），用100维向量表示一个单词
out, h = rnn(x)
print(out.shape, h.shape)

# cell1 = nn.RNNCell(100, 20)
# cell2 = nn.RNNCell(20, 30)
# h1 = torch.zeros(3, 20)
# h2 = torch.zeros(3, 30)
# for xt in x:
#     h1 = cell1(xt, h1)
#     h2 = cell2(h1, h2)
# print(h2.shape)

