import torch.nn as nn
from torchvision import models

def create_mobilenetv2(num_classes):
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    # Replace the last layer with num_classes output units
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model
def create_densenet(num_classes=4):
    # Load DenseNet121 pretrained on ImageNet
    model = models.densenet121(weights="IMAGENET1K_V1")

    # Replace the final classifier to match your number of classes
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model