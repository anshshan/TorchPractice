#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   transfer_learning_tutorial.py
@Time    :   2020/05/20 10:33:43
@Author  :   Shushan An 
@Version :   1.0
@Contact :   shushan_an@cbit.com.cn
@Desc    :   迁移学习的练习，实现蚂蚁和蜜蜂的分类模型
'''
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def im_show(inp, title=None):
    """Imshow the Tensor"""
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def train_model(model, criterion, optimizer, scheduler, num_epochs = 25):
    """用来训练的函数"""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()   # Set model to training mode
            else:
                model.eval()    # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零参数梯度
                optimizer.zero_grad()

                # 前向
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 后向 + 仅在训练阶段进行优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 深度复制mo
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    """一个通用的展示少量预测图片的函数"""
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                im_show(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
    plt.show(fig)




if __name__ == "__main__":
    plt.ion()   # interactive mode

    # 1. 加载数据
    # 在训练数据扩充和归一化
    # 在验证集上仅需归一化

    data_transforms = {
        'train' : transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪一个area然后再resize
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_dir = 'data/hymenoptera_data'

    global dataloaders
    global data_sizes
    global class_names
    global device

    image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    dataloaders = {x : DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x : len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 2. 可视化部分图像数据
    # 获取一批训练数据
    inputs, classes = next(iter(dataloaders['train']))
    # 批量制作网格
    out = torchvision.utils.make_grid(inputs)
    fig = plt.figure()
    im_show(out, title=[class_names[x] for x in classes])
    plt.show(fig)



    # 场景1. 微调ConvNet
    # 加载预训练模型并重置最终完全连接的图层
    model_tf = models.resnet18(pretrained=True)
    num_ftrs = model_tf.fc.in_features
    model_tf.fc = nn.Linear(num_ftrs, 2)

    model_tf = model_tf.to(device)

    criterion = nn.CrossEntropyLoss()

    # 观察所有参数都正在优化
    optimizer_ft = optim.SGD(model_tf.parameters(), lr=0.001, momentum=0.9)

    # 每7个epochs衰减LR通过设置gamma=0.1
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    # 训练和评估模型
    model_tf = train_model(model_tf, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

    visualize_model(model_tf)

    # 场景2. ConvNet作为固定特征提取器
    # 通过requires_grad == Falsebackward()来冻结参数，这样在反向传播的时候不会计算梯度
    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    # 训练和评估模型
    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)

    visualize_model(model_conv)
    