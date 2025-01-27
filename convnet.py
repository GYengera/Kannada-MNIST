import torch
import torch.nn as nn
import torch.nn.functional as F


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.conv1_5 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.conv1_5_bn = nn.BatchNorm2d(num_features=64)
        self.conv1_3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1_3_bn = nn.BatchNorm2d(num_features=64)

        self.conv2_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.conv2_5_bn = nn.BatchNorm2d(num_features=128)
        self.conv2_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(num_features=128)

        self.conv3_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        self.conv3_5_bn = nn.BatchNorm2d(num_features=256)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(num_features=256)

        self.drop1 = nn.Dropout(p=0.2)
        
        self.fc1 = nn.Linear(in_features=512*6*6, out_features=512)
        self.fc1_bn = nn.BatchNorm1d(num_features=512)

        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc2_bn = nn.BatchNorm1d(num_features=256)
        
        self.drop2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc3_bn = nn.BatchNorm1d(num_features=128)

        self.drop3 = nn.Dropout(p=0.2)

        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc4_bn = nn.BatchNorm1d(num_features=64)

        self.out = nn.Linear(in_features=64, out_features=10)


    def forward(self, x):
        x_5 = F.relu(self.conv1_5_bn(self.conv1_5(x))) #(64, 28, 28)
        x_3 = F.relu(self.conv1_3_bn(self.conv1_3(x))) #(64, 28, 28)
        x = F.max_pool2d(torch.cat((x_5, x_3), dim=1), kernel_size=2, stride=2) #(128, 14, 14)

        x_5 = F.relu(self.conv2_5_bn(self.conv2_5(x))) #(128, 14, 14)
        x_3 = F.relu(self.conv2_3_bn(self.conv2_3(x))) #(128, 14, 14)
        x = F.max_pool2d(torch.cat((x_5, x_3), dim=1), kernel_size=2, stride=2) #(256, 7, 7)

        x_5 = F.relu(self.conv3_5_bn(self.conv3_5(x))) #(256, 7, 7)
        x_3 = F.relu(self.conv3_3_bn(self.conv3_3(x))) #(256, 7, 7)
        x = F.max_pool2d(torch.cat((x_5, x_3), dim=1), kernel_size=2, stride=1) #(512, 6, 6)

        x = self.drop1(x)

        x = F.relu(self.fc1_bn(self.fc1(x.view(-1, 6*6*512)))) #(512)
        x = F.relu(self.fc2_bn(self.fc2(x))) #(256)
        x = self.drop2(x)
        x = F.relu(self.fc3_bn(self.fc3(x))) #(128)
        x = self.drop3(x)
        x = F.relu(self.fc4_bn(self.fc4(x))) #(64)
        x = self.out(x) #(10)

        return x
