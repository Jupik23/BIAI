import torch 
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.layer4 = nn.Sequential(
        nn.Dropout(0.3),
        model.layer4
    )
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model