# 名字：李志翔
# 创建时间:2021/8/5 10:54
# CIFAR实战
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from LeNet5 import LeNet5
from ResNet import ResNet18
from torch import nn, optim
import visdom


def main():
    batch_size = 32
    vis = visdom.Visdom()

    CIFAR_train = datasets.CIFAR10('CIFAR', train=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 这里是根据imagenet计算出的方差和均值
                             std=[0.229, 0.224, 0.225]),

    ]), download=True)
    CIFAR_train = DataLoader(CIFAR_train, batch_size=batch_size, shuffle=True)

    CIFAR_test = datasets.CIFAR10('CIFAR', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),                            #这个好像要在前面，不知道为啥
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 这里是根据imagenet计算出的方差和均值
                             std=[0.229, 0.224, 0.225]),

    ]), download=True)
    CIFAR_test = DataLoader(CIFAR_test, batch_size=batch_size, shuffle=True)

    x, label = iter(CIFAR_train).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device("cuda")
    model = ResNet18().to(device)
    print(model)

    criteon = nn.CrossEntropyLoss().to(device)
    optimzer = optim.Adam(model.parameters(), lr=1e-3)

    vis.line([0], [-1], win='loss', opts=dict(title='loss'))
    vis.line([0], [-1], win='acc', opts=dict(title='acc'))

    global_step = 0

    for epoch in range(1000):
        model.train()
        for batchidx, (x, label) in enumerate(CIFAR_train):
            x, label = x.to(device), label.to(device)
            # x  [b 3 32 32]
            # label [b]
            logits = model(x)
            # logits [b 10]
            # label [b]
            # loss tensor scalar
            loss = criteon(logits, label)   # 计算loss

            # backprop
            optimzer.zero_grad()
            loss.backward()   # 反向传播，计算梯度并优化
            optimzer.step()

        #
        print(epoch, loss.item())
        vis.line([loss.item()], [global_step], win='loss', update='append')
        global_step += 1

        model.eval()  # 进行test，关闭dropout
        with torch.no_grad():  # 不需要计算图，更加安全
            # test
            total_correct = 0
            total_num = 0
            for x, label in CIFAR_test:
                x, label = x.to(device), label.to(device)

                # [b,10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)
            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)


if __name__ == '__main__':
    main()
