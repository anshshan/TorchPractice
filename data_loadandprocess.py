#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   data_loadandprocess.py
@Time    :   2020/05/19 11:23:52
@Author  :   Shushan An 
@Version :   1.0
@Contact :   shushan_an@cbit.com.cn
@Desc    :   Pytorch数据加载和处理

安装包:
scikit-image: 用于图像的IO和变换
pandas: 用户更容易地进行csv解析

绝大多数的神经网络都假定图片的尺寸相同。因此一般我们需要进行一些数据的预处理。
    三个转换：
        Rescale：缩放图片
        RandomCrop：对图片随机裁剪
        ToTensor：把numpy格式图片转为torch格式图片（我们需要交换坐标轴）

'''
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform   # 用于图像的IO和变换
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# 忽略警告
import warnings


def show_landmarks(image, landmarks):
    """显示带有地标的图片"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)    # pause a bit so that plots are updated

def show_landmarks_batch(sample_batched):
    """显示一个批次的带有地标的图片"""
    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1,2,0)))
    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size, landmarks_batch[i, :, 1].numpy() + grid_border_size, s=10, marker='.', c='r')
        plt.title('Batch from dataloader')



class FaceLandmarksDataset(Dataset):
    """面部标记数据集"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file(string) : 带注释的csv文件的路径
        root_dir : 包含所有图像的目录
        tansform(callabel, optional) : 一个样本上的可用的可选变换
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """将样本中的图像重新缩放到指定的大小。
    Args:
        output_size(tuple或int) : 所需的输出大小。如果是元组，则输出为与
        output_size匹配。如果是int，则匹配较小的图像边缘到output_size保持纵横比相同。
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 repectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """随机裁剪样本中的图像。

    Args:
        output_size(tuple or int) : 所需输出的大小。如果是int，方形裁剪。
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """将样本中的ndarrays转换为Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # 交换颜色轴
        # numpy包的图片是：H * W * C
        # torch包的图片是：C * H * W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    plt.ion()  # interactive mode
    # 读取数据集
    # landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
    # n = 65
    # img_name = landmarks_frame.iloc[n, 0]
    # landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
    # landmarks = landmarks.astype('float').reshape(-1,2)

    # # 观看部分数据
    # print('Image name: {}'.format(img_name))
    # print('Landmarks shape: {}'.format(landmarks.shape))
    # print('First 4 Landmarks: {}'.format(landmarks[:4]))

    # # 查看样本样例
    # plt.figure()
    # show_landmarks(io.imread(os.path.join('data/faces/', img_name)), landmarks)
    # plt.show()

    # 数据集进行展示
    face_dataset = FaceLandmarksDataset(
        csv_file='data/faces/face_landmarks.csv', root_dir='data/faces/')

    fig = plt.figure()
    for i in range(len(face_dataset)):
        sample = face_dataset[i]
        print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(1, 4, i+1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)

        if i == 3:
            plt.show(fig)
            break

    # 应用图像变换并展示
    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256), RandomCrop(224)])

    # 在样本上应用上述的每个变换
    fig2 = plt.figure()
    sample = face_dataset[65]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transform_sample = tsfrm(sample)

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_landmarks(**transform_sample)
    plt.show(fig2)

    transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces/',
                                               transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))
    
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['landmarks'].size())

        if i == 3:
            break
    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())

        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break