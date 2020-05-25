#!/usr/bin/env python3
# coding:utf-8
'''
Created on 2020-01-16

@author: anshushan

Pytorch 自动微分
    > 通过.requires_grad = True 设置对梯度进行追踪
    > 通过.backward()自动计算所有梯度
    > 使用.detach()停止tensor历史记录追踪
    > 使用with torch.no_grad():包装起来进行，停止tensor历史记录追踪，尤其进行模型评估的时候特别有用
'''
import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

out.backward()
# 这里自动求导的时候如果是标量那么直接使用backward()等同于backward(torch.tensor(1.))
print(x.grad)

# 如果结果不是标量，那么就需要使用雅可比进行求导，表示的是对每个分支求导的权重
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float)
y.backward(v)
print(x.grad)


# 通过.requeire_grad_(...)改变tensor张量的requires_grad标记

a = torch.randn(2,2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# with torch.no_grad():

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)