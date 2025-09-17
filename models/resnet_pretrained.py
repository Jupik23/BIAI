import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
