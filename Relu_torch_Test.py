#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   Relu_torch_Test.py
@Time    :   2020/05/19 15:24:31
@Author  :   Shushan An 
@Version :   1.0
@Contact :   shushan_an@cbit.com.cn
@Desc    :   Pytorch之小试牛刀
'''

import numpy as np
import torch
import random


def network_train_by_numpy():
    """使用numpy包手动实现包含一个隐层的relu网络训练"""

    # N是批量大小； D_in是输入维度；H是隐藏的维度；D_out是输出的维度
    N, D_in, H, D_out = 64, 1000, 100, 10

    # 创建随机输入和输出数据
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    # 随机初始化权重
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

    learning_rate = 1e-6
    for t in range(500):
        # 前向传递：计算预测值y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # 计算和打印损失loss
        loss = np.square(y_pred, y).sum()
        print(t, loss)

        # 方向传播：计算w1和w2对loss的梯度 主要用到了链式法则
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # 更新权重
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


def network_train_by_torch():
    """使用torch张量手动实现一个隐层的relu网络模型的训练"""
    dtype = torch.float
    device = torch.device('cpu')

    # N是批量大小；D_in：是输入维度；H是隐藏的维度；D_out是输出的维度
    N, D_in, H, D_out, = 64, 1000, 100, 10

    # 创建随机输入和输出的数据
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # 随机初始化权重
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(500):
        # 前向传递：计算预测y
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

        # 计算和打印损失
        loss = (y_pred - y).pow(2).sum().item()
        print(t, loss)

        # 反向传播：计算w1和w2相对于损失的梯度
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        # 使用梯度更新权重值
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


def auto_network_train_by_torch():
    dtype = torch.float
    device = torch.device('cpu')

    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(500):
        # 前向传播：在tensor上进行预测值的计算
        # 由于这里设置了梯度的记录，因此不需要中间值来计算梯度
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        # 在tensor上计算损失和打印
        loss = (y_pred - y).pow(2).sum()
        print(t, loss)

        loss.backward()

        # 这里我们直接对w1和w2进行原地操作，因此不需要构建计算图来保存梯度，所以使用torch.no_grad()
        with torch.no_grad():

            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            # 反向传播后将梯度手动清零
            w1.grad.zero_()
            w2.grad.zero_()


class MyReLu(torch.autograd.Function):
    """
    我们可以通过建立torch.autograd的子类来实现我们自定义的autograd函数，
    并完成张量的正向和反向传播
    """
    @staticmethod
    def forward(ctx, x):
        """
        在正向传播中，我们接收一个上下文对象和一个包含输入的张量；
        我们必须返回一个包含输出的张量，
        并且我们可以使用上下文对象来缓存对象，以便在反向传播中使用。
        """
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        在反向传播中，我们接收一个上下文对象和一个张量；
        其中包含了相对于正向传播过程中产生的输出的损失的梯度
        我们可以从上下文对象中检索缓存的数据
        并且必须计算并返回与正向传播的输入相关的损失的梯度
        """
        x, = ctx.save_tensors
        grad_x = grad_output.clone()
        grad_x[x < 0] = 0
        return grad_x

class DynamicNet(torch.nn.Module):
    
    def __init__(self, D_in, D_out):
        """
        在构造函数中，我们构造了三个nn.Linear实例，它们将在前向传播时被使用
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H,H)
        self.output_linear = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        """
        对于模型的前向传播，我们随机选择0、1、2、3，
        并重用了多次计算隐藏层的middle_linear模块。
        由于每个前向传播构建一个动态计算图，
        我们可以在定义模型的前向传播时使用常规Python控制流运算符，如循环或条件语句。
        在这里，我们还看到，在定义计算图形时多次重用同一个模块是完全安全的。
        这是Lua Torch的一大改进，因为Lua Torch中每个模块只能使用一次。
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred

def network_train_by_torch_define_gradfunction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # N是批大小； D_in 是输入维度；
    # H 是隐藏层维度； D_out 是输出维度
    N, D_in, H, D_out = 64, 1000, 100, 10

    # 产生输入和输出的随机张量
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)

    # 产生随机权重的张量
    w1 = torch.randn(D_in, H, device=device, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, requires_grad=True)

    learning_rate = 1e-6
    for t in range(500):
        # 正向传播：使用张量上的操作来计算输出值y；
        # 我们通过调用 MyReLU.apply 函数来使用自定义的ReLU
        y_pred = MyReLU.apply(x.mm(w1)).mm(w2)

        # 计算并输出loss
        loss = (y_pred - y).pow(2).sum()
        print(t, loss.item())

        # 使用autograd计算反向传播过程。
        loss.backward()

        with torch.no_grad():
            # 用梯度下降更新权重
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            # 在反向传播之后手动清零梯度
            w1.grad.zero_()
            w2.grad.zero_()

def network_train_by_torch_same_weight():
    """动态图的共享权重"""
    # N是批大小；D是输入维度
    # H是隐藏层维度；D_out是输出维度
    N, D_in, H, D_out = 64, 1000, 100, 10

    # 产生输入和输出随机张量
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    # 实例化上面定义的类来构造我们的模型
    model = DynamicNet(D_in, H, D_out)

    # 构造我们的损失函数（loss function）和优化器（Optimizer）。
    # 用平凡的随机梯度下降训练这个奇怪的模型是困难的，所以我们使用了momentum方法。
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    for t in range(500):

        # 前向传播：通过向模型传入x计算预测的y。
        y_pred = model(x)

        # 计算并打印损失
        loss = criterion(y_pred, y)
        print(t, loss.item())

        # 清零梯度，反向传播，更新权重 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()