import torch
import torch.nn as nn
from torchvision.models import resnet18


class Identity(nn.Module):
    def forward(self, x):
        return x


class ResNet18Client(nn.Module):
    """
    Client-side submodel of ResNet-18.
    Supported cuts:
        C1: after stem
        C2: after layer1
        C3: after layer2
        C4: after layer3
    """
    def __init__(self, cut_id: str):
        super().__init__()
        self.cut_id = cut_id

        base = resnet18(num_classes=10)

        stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )

        if cut_id == "C1":
            self.model = nn.Sequential(
                stem
            )
        elif cut_id == "C2":
            self.model = nn.Sequential(
                stem,
                base.layer1,
            )
        elif cut_id == "C3":
            self.model = nn.Sequential(
                stem,
                base.layer1,
                base.layer2,
            )
        elif cut_id == "C4":
            self.model = nn.Sequential(
                stem,
                base.layer1,
                base.layer2,
                base.layer3,
            )
        else:
            raise ValueError(f"Unsupported cut_id: {cut_id}")

    def forward(self, x):
        return self.model(x)


class ResNet18Server(nn.Module):
    """
    Server-side tail model of ResNet-18.
    """
    def __init__(self, cut_id: str, num_classes: int = 10):
        super().__init__()
        self.cut_id = cut_id

        base = resnet18(num_classes=num_classes)

        tail_layers = []

        if cut_id == "C1":
            tail_layers.extend([
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
                base.avgpool,
            ])
            self.feature_tail = nn.Sequential(*tail_layers)
            self.fc = base.fc

        elif cut_id == "C2":
            tail_layers.extend([
                base.layer2,
                base.layer3,
                base.layer4,
                base.avgpool,
            ])
            self.feature_tail = nn.Sequential(*tail_layers)
            self.fc = base.fc

        elif cut_id == "C3":
            tail_layers.extend([
                base.layer3,
                base.layer4,
                base.avgpool,
            ])
            self.feature_tail = nn.Sequential(*tail_layers)
            self.fc = base.fc

        elif cut_id == "C4":
            tail_layers.extend([
                base.layer4,
                base.avgpool,
            ])
            self.feature_tail = nn.Sequential(*tail_layers)
            self.fc = base.fc

        else:
            raise ValueError(f"Unsupported cut_id: {cut_id}")

    def forward(self, x):
        x = self.feature_tail(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_split_models(cut_id: str, num_classes: int = 10):
    client_model = ResNet18Client(cut_id)
    server_model = ResNet18Server(cut_id, num_classes=num_classes)
    return client_model, server_model