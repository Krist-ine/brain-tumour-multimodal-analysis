import torch
import torch.nn as nn
from torchvision import models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNetClassifier, self).__init__()

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        # Freeze early layers (for small datasets)
        for param in list(self.resnet.parameters())[:100]:
            param.requires_grad = False

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


if __name__ == '__main__':
    model = ResNetClassifier(num_classes=4)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
