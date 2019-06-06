import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Input = 1 * 28 * 28
        # Output = 16 * 28 * 28
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(16)
        # Input = 16 * 28 *28
        # Output = 16 * 28 * 28
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        # Input = 16 * 28 * 28
        # Output = 16 * 28 *28
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(16)
        # Input = 16 * 28 * 28
        # Output = 16 * 14 * 14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input = 16 * 14 * 14
        # Output = 32 * 14 * 14
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.batch4 = nn.BatchNorm2d(32)
        # Input = 32 * 14 * 14
        # Output = 32 * 14 * 14
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.batch5 = nn.BatchNorm2d(32)
        # Input = 32 * 14 * 14
        # Output = 32 * 14 *14
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.batch6 = nn.BatchNorm2d(32)
        # Input = 32 * 14 *14
        # Output = 32 * 7 * 7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input = 32 * 7 * 7
        # Output = 48 * 5 * 5
        self.conv7 = nn.Conv2d(32, 48, kernel_size=3)
        self.batch7 = nn.BatchNorm2d(48)

        # Input = 48 * 5 * 5
        # Output= 32 * 5 * 5
        self.conv8 = nn.Conv2d(48, 32, kernel_size=1)
        self.batch8 = nn.BatchNorm2d(32)

        # Input = 32 * 5 * 5
        # Output= 16 * 5 * 5
        self.conv9 = nn.Conv2d(32, 16, kernel_size=1)
        self.batch9 = nn.BatchNorm2d(16)

        self.pool3 = nn.AvgPool2d(5)

        self.fc1 = nn.Linear(1 * 1 * 16, 10)




    def forward(self,x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv4(x)
        x = self.batch4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.batch5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.batch6(x)
        x = F.relu(x)

        x = self.pool2(x)

        x = self.conv7(x)
        x = self.batch7(x)
        x = F.relu(x)

        x = self.conv8(x)
        x = self.batch8(x)
        x = F.relu(x)

        x = self.conv9(x)
        x = self.batch9(x)
        x = F.relu(x)

        x = self.pool3(x)

        x = x.reshape(-1, 16)
        x = self.fc1(x)
        return x



