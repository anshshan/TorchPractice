#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   neural_networks_tutorial.py
@Time    :   2020/05/18 14:39:52
@Author  :   Shushan An 
@Version :   1.0
@Contact :   shushan_an@cbit.com.cn
@Desc    :   神经网络通过torch.nn包

神经网络训练的过程包含以下几点：
    > 定义一个包含可训练参数的神经网络      torch.nn
    > 迭代整个输入
    > 通过神经网络处理输入      forward()
    > 计算损失（loss）      torch.nn中的损失函数
    > 反向传播梯度到神经网络的参数      zero_grad()情况缓存的梯度, backgrad()
    > 更新网络的参数，典型的用一个简单的更新方法： weight = weight - learning_rate * gradient       可以是同torch.optim中的参数更新函数进行更新
'''
# 定义神经网络
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5*5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 进行展开
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# 获取模型的可训练参数
params = list(net.parameters())
print(len(params))
print(params[0].size()) # con1v's .weight

input = torch.randn(1,1,32,32)
out = net(input)
print(out)

# 所有参数梯度缓存器清零，用随机的梯度来反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))

# 计算损失值
    # 损失函数
output = net(input)
target = torch.randn(10)
target = target.view(1, -1) # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 通过python来实现梯度更新
# learning.rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)

# 或者使用torch.optim包进行
import torch.optim as optim
#  create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in you training loop
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # Does the update