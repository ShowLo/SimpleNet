# -*- coding: UTF-8 -*-

'''
简单的用于分类的网络
'''

import torch

import torch.nn as nn

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        # 三个卷积层用于提取特征
        # 1 input channel image 90x90, 8 output channel image 44x44
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 8 input channel image 44x44, 16 output channel image 22x22
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 16 input channel image 22x22, 32 output channel image 10x10
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 分类
        self.classifier = nn.Sequential(
            nn.Linear(32 * 10 * 10, 3)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 32 * 10 * 10)
        x = self.classifier(x)
        return x