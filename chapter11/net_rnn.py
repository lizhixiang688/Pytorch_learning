# 名字：李志翔
# 创建时间:2021/8/7 11:08
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(
            input_size=1,      # 不需要embedding，一个数据点就是一个数字
            hidden_size=16,    # h的维度
            num_layers=1,
            batch_first=True   # batch 在前面
        )
        self.linear = nn.Linear(16, 1)  # 输出层  16:hidden_size  1:output_size

    def forward(self, x, hidden_prev):   # x =>[1, 49, 1]  1个batch 49个点（sequence） 不需要embedding,所以为1
        out, hidden_prev = self.rnn(x, hidden_prev)  # [1, 49, 1]=>[1, 49, 16]
        out = out.view(-1, 16)      # [1,seq,h] => [seq,h]
        out = self.linear(out)      # [seq,h] => [seq,1]
        out = out.unsqueeze(dim=0)  # [seq,1] => [1,seq,1]   因为要和label比较loss，所以维度要一样
        return out, hidden_prev


if __name__ == '__main__':
    model = Net()
    x = torch.randn(1, 49, 1)
    hidden = torch.zeros(1, 1, 16)
    out, hidden = model(x, hidden)
    print(out.shape, hidden.shape)
