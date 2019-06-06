import torch.nn as nn
import torch.nn.functional


class ConvNet(nn.Module):
    def __init__(self):
        super.__init__(self)
        self.net = nn.Sequential(
            
            nn.Conv2d(1,16,3),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 48, 3),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, 3),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, 3),
            nn.BatchNorm2d(48),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 64, 3),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 48, 1),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 32, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),


            nn.AvgPool2d(5),
            nn.Linear(16,10)

        )


    def forward(self,x):
        return self.net.forward(x)



