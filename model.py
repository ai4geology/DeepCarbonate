# model.py
import torch.nn as nn
from torchvision.models import resnet18, resnet50, vgg16, densenet121, mobilenet_v2, efficientnet_b0

MODEL_MAP = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'vgg16': vgg16,
    'densenet121': densenet121,
    'mobilenet_v2': mobilenet_v2,
    'efficientnet_b0': efficientnet_b0
}

def create_model(model_name, num_classes):
    """Create an untrained base model"""
    constructor = MODEL_MAP[model_name]
    model = constructor(pretrained=False)  # Without pre training
    
    # Modify classification header
    if 'resnet' in model_name:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'vgg' in model_name:
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'densenet121':
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif 'efficientnet' in model_name:
        model.classifier = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model