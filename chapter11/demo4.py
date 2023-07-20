# 名字：李志翔
# 创建时间:2021/8/7 11:24
# 时间序列预测实战

import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from net_rnn import Net

num_time_steps = 50
lr = 0.01

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)

hidden_prev = torch.zeros(1, 1, 16)  # h0

for iter in range(6000):
    # 给train提供数据
    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = torch.tensor(data[:-1]).float().view(1, 49, 1)  # 1个batch 49个点 1个维度来表示（不需要embedding）
    y = torch.tensor(data[1:]).float().view(1, 49, 1)   # label

    output, hidden_prev = model(x, hidden_prev)
    hidden_prev = hidden_prev.detach()

    loss = criterion(output, y)    # 维度都是[1,49,1]
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print('Iteration:{} loss {}'.format(iter, loss.item()))

# 这里是给test提供数据
start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, 49, 1)
y = torch.tensor(data[1:]).float().view(1, 49, 1)

# 这里进行test
predictions = []
input = x[:, 0, :]
for _ in range(x.shape[1]):
    input = input.view(1, 1, 1)
    (pred, hidden_prev) = model(input, hidden_prev)
    input = pred
    predictions.append(pred.detach().numpy().ravel()[0])

"""
------------------下面是画图的部分----------------------
"""
x = x.data.numpy().ravel()
y = y.data.numpy()
plt.scatter(time_steps[:-1], x.ravel(), s=90)
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[1:], predictions)
plt.show()
