# 名字：李志翔
# 创建时间:2021/8/2 11:35
# 划分数据集

import torch.utils.data.dataloader
from torchvision import datasets, transforms

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                       , transforms.Normalize((0.1307,), (0.3081,))])), batch_size=200, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([transforms.ToTensor()
                                                    , transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=200, shuffle=True
)

train_db, val_db = torch.utils.data.random_split(train_loader.dataset, [50000, 10000])
train_loader = torch.utils.data.DataLoader(
    train_db, batch_size=200, shuffle=True)
print(len(train_loader.dataset))
