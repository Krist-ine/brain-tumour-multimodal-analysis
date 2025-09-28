# src/models/cnn.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ImageCNN(nn.Module):
    def __init__(self, num_classes=4, embed_dim=128):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.embed_layer = nn.Linear(num_features, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embed_layer(features)
        logits = self.classifier(embedding)
        return logits, embedding