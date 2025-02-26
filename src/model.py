import torch.nn as nn
import torch

class QuickDraw(nn.Module):
    def __init__(self, num_classes):
        super(QuickDraw, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # Increased filters to 32
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        # Calculate the correct input size for the fully connected layer
        self.fc = nn.Linear(13 * 13 * 32, num_classes)  # Adjusted input size

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x