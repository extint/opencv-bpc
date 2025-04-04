import torch
import torch.nn as nn
import torchvision.models as tv_models

class SimplePoseNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = tv_models.resnet50(
            weights=(tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        )
        layers = list(backbone.children())[:-1]  # Remove the classification head
        self.backbone = nn.Sequential(*layers)
        self.fc = nn.Linear(2048, 5)  # Output: [Rx, Ry, Rz, cx, cy]

    def forward(self, x):
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        return self.fc(feats)
