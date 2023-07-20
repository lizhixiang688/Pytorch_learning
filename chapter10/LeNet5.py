# 名字：李志翔
# 创建时间:2021/8/5 11:10

import torch
from torch import nn
from torch.nn import functional as F


class LeNet5(nn.Module):
    """
    for CIFAR-10 dataset.
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x:[b,3,32,32]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
        )
        # flatten
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        # [b 3 32 32]
        temp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(temp)
        # [b 16 5 5]
        print('conv_out:', out.shape)

        #
        # self.criteon = nn.CrossEntropyLoss()

    def forward(self, x):
        """

        :param x:[b 3 32 32]
        :return:
        """
        batchsize = x.size(0)
        # [b 3 32 32] => [b 16 5 5]
        x = self.conv_unit(x)
        # [b 16 5 5] => [b 16*5*5]
        x = x.view(batchsize, 16 * 5 * 5)
        # [b 16*5*5] => [b 10]
        logits = self.fc_unit(x)

        # [b 10]
        # pred=F.softmax(logits,dim=1),直接使用crossentropy
        # loss = self.criteon(logits, y)

        return logits


def main():
    net = LeNet5()

    temp = torch.randn(2, 3, 32, 32)
    out = net(temp)
    # [b 16 5 5]
    print('lenet_out:', out.shape)


if __name__ == '__main__':
    main()
