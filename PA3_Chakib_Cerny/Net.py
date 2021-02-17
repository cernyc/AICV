import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, mode):
        super(Net, self).__init__()
        #convolutional layers using a 5 * 5 kernel
        self.conv1 = nn.Conv2d(3, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.conv3 = nn.Conv2d(16, 26, 5, 1)
        self.conv4 = nn.Conv2d(26, 36, 5, 1, 1)
        self.conv5 = nn.Conv2d(36, 46, 5, 1)
        self.conv6 = nn.Conv2d(46, 56, 5, 1)

        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers that takes different inputs based on the convolutional layers and output 10 after the 3rd for our 10 classes
        self.fc1 = nn.Linear(26 * 3 * 3, 120)
        self.fc12 = nn.Linear(36 * 3 * 3, 120)
        self.fc13 = nn.Linear(56 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        else:
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)

    def model_1(self, x):

        x = F.relu(self.conv1(x))
       # print(x.shape)
        x = F.max_pool2d(x, 2)
       # print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        #x = F.max_pool2d(x, 2)
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = F.max_pool2d(x, 2)
        #print(x.shape)
        x = x.view(-1, 26 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def model_2(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        # x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.conv4(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = x.view(-1, 36 * 3 * 3)
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # Use two convolutional layers.
    def model_3(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        # x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        # x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.conv3(x))
        #x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.conv4(x))
        # print(x.shape)
        x = F.relu(self.conv5(x))
        # print(x.shape)
        x = F.relu(self.conv6(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2)
        #print(x.shape)
        x = x.view(-1, 56 * 5 * 5)
        x = F.relu(self.fc13(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

