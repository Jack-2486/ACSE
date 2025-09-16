import torch.nn as nn
import torch
from AlexNet.CBAM import CBAMLayer
from AlexNet.SENet import SENet

class AlexNet(nn.Module):
    # 类AlexNet继承nn.module这个父类
    def __init__(self, num_classes=7, init_weights=False):  # num_classes是指输出的图片种类个数，init_weights=False意味着不定义模型中的初始权重
        # 通过初始化函数，定义网络在正向传播过程中需要使用的层结构
        super(AlexNet, self).__init__()
        # 将专门用于提取图像特征的结构的名称取为features
        self.features = nn.Sequential(  # nn.Sequential模块，可以将一系列的层结构进行打包，组合成一个新的结构，
            nn.Conv2d(3, 48, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),  # 使用Relu激活函数时要设置inplace=True
            SENet(48),
            CBAMLayer(48),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SENet(128),
            CBAMLayer(128),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            SENet(192),
            CBAMLayer(192),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SENet(128),
            CBAMLayer(128),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(p=0.2),
            # Conv d卷积层  BatchNorm 全连接层
        )
        # 将三个全连接层打包成一个新的模块，分类器
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(128 * 2 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        # 是否初始化权重，如果初始化函数中的init_weights=Ture,就会进入到初始化权重的函数

    def forward(self, x):
        # forward正向传播过程，x为输入的变量
        x = self.features(x)
        # 将训练样本输入features
        # print("Final feature map shape:", x.shape)  # 预期输出：[32, 256, 6, 6]
        # print(x.shape)  # 调试用
        x = torch.flatten(x, start_dim=1)
        # 将输入的变量进行展平从深度高度宽度三个维度进行展开，索引从1开始，展成一个一维向量
        x = self.classifier(x)
        # 将展平后的数据输入到分类器中（三个全连接层组成的）
        return x  # 最后的输出为图片类别

    # 初始化权重的函数
    def _initialize_weights(self):
        for m in self.modules():  # 遍历self.modules这样一个模块，该模块继承自它的父类nn.module，该模块会迭代每一个层次
            if isinstance(m, nn.Conv2d):  # 如果该层次是一个卷积层，就会使用kaiming_normal_这样一个方法初始化权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 如果偏值不为空的话，就用0对权重进行初始化
            elif isinstance(m, nn.Linear):  # 如果该层次是一个全连接层，就用normal进行初始化
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布对权重进行赋值，均值为0，方差为0.01
                if m.bias is not None:  # 检查m.bias是否为None
                    nn.init.constant_(m.bias, 0)  # 设置全连接层的偏值为0
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)  # 通常将BN层的权重初始化为1
                nn.init.constant_(m.bias, 0)   # 通常将BN层的偏置初始化为0



