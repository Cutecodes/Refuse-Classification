import torch
import torch.nn as nn




class CNN(nn.Module):
    """
    网络模型
    """
    def __init__(self, image_size, num_classes):
        super(CNN, self).__init__()
        # conv1: Conv2d -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        # conv2: Conv2d -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        # fully connected layer
        self.dp1 = nn.Dropout(0.20)
        self.fc1 = nn.Linear(25088, 4096)
        self.dp2 = nn.Dropout(0.10)
        self.fc2 = nn.Linear(4096, 256)
        self.dp3 = nn.Dropout(0.10)
        self.fc3 = nn.Linear(256,num_classes)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        """
        input: batchsize * 3 * image_size * image_size
        output: batchsize * num_classes
        """
        x = self.conv1(x)
        
        x = self.conv2(x)
        
        x = self.conv3(x)
        
        x = self.conv4(x)
        
        # view(x.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        x = x.view(x.size(0), -1)
        
        x = self.dp1(x)
        x = self.fc1(x)
        x = self.ReLU(x)
        
        x = self.dp2(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        
        x = self.dp3(x)
        x = self.fc3(x)
        
        output = self.ReLU(x)

        return output
