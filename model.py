"""Cassava classifiers and checkpoint loading."""

import torch
from torch import nn
from torchvision import models


def build_model(name, num_classes=5, pretrained=False):
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name != "cnn":
        raise ValueError(f"Unknown model: {name}")
    layers = []
    for incoming, outgoing in ((3, 16), (16, 32), (32, 64)):
        layers += [
            nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ]
    return nn.Sequential(
        *layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes)
    )


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, target_device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["architecture"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(target_device).eval(), checkpoint
