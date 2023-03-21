import torch
import torch.nn as nn
from torchvision import transforms


class ChessCNNv3(nn.Module):
    def __init__(self):
        super(ChessCNNv3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(10816, 64)
        self.fc2 = nn.Linear(64, 13)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, 0.2, self.training)
        x = self.fc2(x)
        return x

def load_model(path: str):

    model_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    model = torch.load(path)

    return model, model_transform
