import torch.nn as nn
import torchvision.models as models

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_p=0.5):
        super(SimpleCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model(version, dropout_p=0.5, num_classes=2):
    if version == 1:
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif version == 2:
        model = models.resnet18(pretrained=False)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif version == 3:
        model = models.resnet18(pretrained=False)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif version == 4:
        model = models.resnet18(pretrained=False)
        for param in model.parameters():
            param.requires_grad = False
        model.layer4 = nn.Sequential(
            nn.Dropout(p=dropout_p),
            model.layer4
        )
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif version == 5:
        model = SimpleCNN(num_classes=num_classes, dropout_p=dropout_p)
    else:
        raise ValueError("Invalid model version selected (choose 1–5)")
    return model
