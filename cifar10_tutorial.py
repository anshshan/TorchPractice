#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   cifar10_tutorial.py
@Time    :   2020/05/18 15:49:57
@Author  :   Shushan An 
@Version :   1.0
@Contact :   shushan_an@cbit.com.cn
@Desc    :   Pytorch图像分类器

训练一个图像分类器步骤：
    1. 使用torchvision加载并归一化CIFAR10的训练和测试数据集
    2. 定义一个卷积神经网络
    3. 定义一个损失函数
    4. 在训练样本数据上训练网络
    5. 在测试样本数据上测试网络
'''
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    '''神经网络模型
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
    '''Function to show an image
    '''
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":

    # 1. 加载数据并进行归一化处理
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 2. 展示一些训练的照片
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print  labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # 3. 定义卷积神经网络模型
    net = Net()


    # 4. 定义损失函数和参数更新的方式
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Add如何在GPU上进行训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(device)
    # 将网络模型和数据都转换成cuda张量
    net.to(device)

    # 5. 在训练样本上训练数据
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # 转换成cuda张量
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # 这里我们给出的是每2000个batch输出一次结果
            running_loss += loss
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %(epoch + 1), i + 1, running_loss / 2000)
                running_loss = 0.0

    print('Finished Training')

    # 6. 在测试数据集上进行测试
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            # torch.max(inputs, dim),dim=1,返回每行的最大值；dim=0,返回每列的最大值
            _, pridicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pridicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, pridicted = torch.max(outputs, 1)
            c = (pridicted == labels).squeeze()
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
