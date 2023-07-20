# 名字：李志翔
# 创建时间:2021/8/4 17:49
#数据增强(data argumentation)


import torch.utils.data.dataloader
from torchvision import datasets,transforms



train_loader=torch.utils.data.DataLoader(
    datasets.MNIST('../data',train=True,download=True,
                   transform=transforms.Compose([
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomVerticalFlip(),
                       transforms.RandomRotation(15),
                       transforms.Resize([32,32]),
                       transforms.RandomCrop([28,28]),
                       transforms.ToTensor()
                  ,])),batch_size=200,shuffle=True
)

print(len(train_loader.dataset))    #还是60000个。。。

