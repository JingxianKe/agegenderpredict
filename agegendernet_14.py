from torch import nn, optim

class AgeGenderNet(nn.Module):
    def __init__(self, num_classes):
        super(AgeGenderNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25))

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25))

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(384),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25))

        self.fc1 = nn.Sequential(
            nn.Linear(13824, 512),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc3 = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

