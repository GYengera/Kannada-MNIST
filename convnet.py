import torch
import torch.nn as nn
import torch.nn.functional as F

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(num_features=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=128)

        self.fc1 = nn.Linear(in_features=128*7*7, out_features=1024)
        self.fc1_bn = nn.BatchNorm1d(num_features=1024)

        self.out = nn.Linear(in_features=1024, out_features=10)


    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2) #(1, 14, 14) write the dimensions

        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2) #(1, 7, 7)

        x = F.relu(self.fc1_bn(self.fc1(x.view(-1, 7*7*128))))
        x = self.out(x)

        return x
