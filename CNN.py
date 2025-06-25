import torch
import torch.nn as nn
import torch.nn.functional as F


class BreakHisCNN(nn.Module):
    """
    基于 PyTorch 的简单卷积神经网络，用于 BreakHis 数据集的特征提取与分类。
    输入图像尺寸为 (3, 350, 230)。

    参数:
        nn.Moudle

    返回:
        out: 输出结果
    """

    def __init__(self, num_classes: int = 2):
        super(BreakHisCNN, self).__init__()
        # 卷积块1: Conv -> ReLU -> Pool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        # 卷积块2: Conv -> ReLU -> Pool
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)

        # 卷积块3: Conv -> ReLU -> Pool
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1)

        # 全局平均池化（用于特征提取）
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接分类层
        self.fc1 = nn.Linear(128, 64)  # FC-1: 64 节点
        self.bn1 = nn.BatchNorm1d(64)  # BatchNorm

        self.fc2 = nn.Linear(64, 64)  # FC-2: 64 节点
        self.bn2 = nn.BatchNorm1d(64)  # BatchNorm

        self.fc3 = nn.Linear(64, num_classes)  # FC-3: num_classes 输出

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 卷积+池化
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # 全局平均池化并展平
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # 隐藏全连接层 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # 隐藏全连接层 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # 输出层 + Softmax
        out = self.fc3(x)
        return out
