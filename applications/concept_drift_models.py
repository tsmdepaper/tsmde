import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

class MyDataLoader(torch.utils.data.Dataset):

    def __init__(self, fnames, labels, transforms):

        super().__init__()

        self.fnames = fnames
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
            
        fname = self.fnames[idx]
        label = self.labels[idx]

        # load image
        img = Image.open(fname)

        return self.transforms(img), label

class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)
        
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    # forward pass to be used in actual training
    def forward(self, x):
        x = F.leaky_relu(self.maxpool1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool3(self.conv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool4(self.conv4(x)), 0.2, inplace=True)

        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        x = self.fc3(x)

        return torch.sigmoid(x)

    # forward pass to get features out
    def fea(self, x):
        x = F.leaky_relu(self.maxpool1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool3(self.conv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.maxpool4(self.conv4(x)), 0.2, inplace=True)

        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        
        return self.fc2(x) 