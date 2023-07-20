# 名字：李志翔
# 创建时间:2021/8/7 17:48
# LSTM和LXTMCell

import torch
import torch.nn as nn

lstm = nn.LSTM(input_size=100, hidden_size=20, num_layers=4)
x = torch.randn(10, 3, 100)   # 这里3代表3个句子（batch） 10个单词，100的embedding
out, (h, c) = lstm(x)
print(out.shape, h.shape, c.shape)

# cell1 = nn.LSTMCell(input_size=100, hidden_size=30)
# cell2 = nn.LSTMCell(input_size=30, hidden_size=20)
# h1 = torch.zeros(3, 30)
# c1 = torch.zeros(3, 30)
# h2 = torch.zeros(3, 20)
# c2 = torch.zeros(3, 20)
# for xt in x:
#     h1, c1 = cell1(xt, [h1, c1])  # 注意这里的xt和h1
#     h2, c2 = cell2(h1, [h2, c2])
# print(h2.shape, c2.shape)
