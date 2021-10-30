'''
Project - ICASSP PAPER
Author  - Maosen Wang
Date    - 2021/10/01
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd

# Tensor： 就像ndarray一样,一维Tensor叫Vector，二维Tensor叫Matrix，三维及以上称为Tensor

"""创建一个转换器，将torchvision数据集的输出范围[0,1]转换为归一化范围的张量[-1,1]"""
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 在小型数据集上，通过随机水平翻转来实现数据增强
    transforms.RandomGrayscale(),  # 将图像以一定的概率转换为灰度图像
    transforms.ToTensor(),  # 数据集加载时，默认的图片格式是 numpy，所以通过 transforms 转换成 Tensor,图像范围[0, 255] -> [0.0,1.0]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,
                                           0.5))])  # 使用公式进行归一化channel=（channel-mean）/std，因为transforms.ToTensor()已经把数据处理成[0,1],那么(x-0.5)/0.5就是[-1.0, 1.0]

"""下载训练数据和测试数据"""
trainset = torchvision.datasets.CIFAR10('data', train=True,
                                        download=True,
                                        transform=transform)
testset = torchvision.datasets.CIFAR10('data', train=False,
                                       download=True,
                                       transform=transform)

"""创建训练加载器和测试加载器，DataLoader 从数据集中不断提取数据然后送往 Model 进行训练和预测"""
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True,
                                          num_workers=2)  # 加载数据的时候使用几个子进程
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=True,
                                         num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""创建网络"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # 建立第一个卷积层
            nn.Conv2d(  # 二维卷积层，通过过滤器提取特征
                in_channels=3,  # 图片有几个通道（灰度图片1通道，彩色图片3通道）
                out_channels=16,  # 过滤器也就是卷积核的个数（每一个卷积核都是3通道，和输入图片通道数相同，但是输出的个数依然是卷积核的个数，是因为运算过程中3通道合并为一个通道）
                kernel_size=5,  # 过滤器的宽和高都是5个像素点
                stride=1,  # 每次移动的像素点的个数（步子大小）
                padding=2,  # 在图片周围添加0的层数，stride=1时，padding=(kernel_size-1)/2
            ),  # (3,32,32)-->(16,32,32)
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2),  # 池化层，压缩特征，一般采用Max方式，kernel_size=2代表在2*2的特征区间内去除最大的) (16,32,32)-->(16,16,16)
        )
        self.conv2 = nn.Sequential(  # 建立第二个卷积层
            nn.Conv2d(16, 32, 5, 1, 2),  # (16,16,16) -->(32,16,16)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32,16,16)-->(32,8,8)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),  # (32,8,8)-->(64,8,8)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (64,8,8)-->(64,4,4)
        )
        self.out = nn.Linear(64 * 4 * 4, 10)  # 全连接层，矩阵相乘形成（1，10）

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64 * 4 * 4, )  # 行数-1代表不知道多少行的情况下，根据原来Tensor内容和Tensor的大小自动分配行数，但是这里为1行
        output = self.out(x)
        return output


if __name__ == '__main__':
    saveresults = [[], []]


    net = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 使用Adam优化器，学习率为0.001
    noise = torch.zeros(100, 3072, device=device, requires_grad=False)
    noise = noise.reshape(100, 3, 32, 32)

    for epoch in range(50):  # 迭代训练10次
        correct = 0
        total = 0
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)  # 计算损失
            predicted = torch.max(outputs, 1)[1].data.cpu().numpy()
            total += labels.size(0)
            correct += (predicted == labels.data.cpu().numpy()).sum()
            optimizer.zero_grad()  # 梯度先全部降为0
            loss.backward()  # 反向传递过程
            optimizer.step()  # 以学习效率0.001来优化梯度
        print('train accuracy: %d%%' % (100 * correct / total))
        train_acc = correct / total
        saveresults[0].append(train_acc)
        test_correct = 0
        test_total = 0
        for i, data in enumerate(testloader):  # 迭代并返回下标和数据

            images, labels = data
            noise = noise.to(device)
            images = images.to(device)
            labels = labels.to(device)
            images = images + noise
            outputs = net(images)
            predicted = torch.max(outputs, 1)[1].data.cpu().numpy()
            test_total += labels.size(0)
            test_correct += (predicted == labels.data.cpu().numpy()).sum()
        print('test accuracy: %d%%' % (100 * test_correct / test_total))
        test_acc = test_correct / test_total
        saveresults[1].append(test_acc)

    print('begin test')

    for iii in range(20):
        value = (iii+1) * 0.3
        noise = torch.normal(0, value, (100, 3072), device=device, requires_grad=False)
        #noise = torch.zeros(100,3072)
        noise = noise.reshape(100, 3, 32, 32)
        print('noise=')
        print(value)
        print('noise=')

        for i, data in enumerate(testloader):  # 迭代并返回下标和数据
            images, labels = data
            noise = noise.to(device)
            images = images.to(device)
            labels = labels.to(device)
            images = images + noise
            outputs = net(images)
            predicted = torch.max(outputs, 1)[1].data.cpu().numpy()
            test_total += labels.size(0)
            test_correct += (predicted == labels.data.cpu().numpy()).sum()
        print('test accuracy: %d%%' % (100 * test_correct / test_total))
        test_acc = test_correct / test_total
        saveresults[0].append(test_acc)
        saveresults[1].append(test_acc)





    name = ['Train Accuracy', 'Test Accuracy']
    result = pd.DataFrame(list(zip(saveresults[0], saveresults[1])), columns=name)
    result.to_csv('Results.csv')

