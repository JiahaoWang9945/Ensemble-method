import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import densenet121
from torchvision.models import mobilenet_v3_large


class Ensemble_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.resnet = resnet18(pretrained=True)
        self.densenet = densenet121(pretrained=True)
        self.mobilenet = mobilenet_v3_large(pretrained=True)

        del self.resnet.fc

        del self.densenet.classifier

        del self.mobilenet.classifier

        self.fc = nn.Linear(3584, num_classes)

    def forward(self, x):
        x1 = self.resnet(x)
        x2 = self.densenet(x)
        x3 = self.mobilenet(x)

        out = torch.cat((x1, x2, x3), dim=1)

        out = self.fc(out)

        return out


class Ensemble_r18_d121(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.resnet = resnet18(pretrained=True)
        self.densenet = densenet121(pretrained=True)

        del self.resnet.fc

        del self.densenet.classifier

        self.fc = nn.Linear(3072, num_classes)

    def forward(self, x):
        x1 = self.resnet(x)
        x2 = self.densenet(x)

        out = torch.cat((x1, x2), dim=1)

        out = self.fc(out)

        return out


class Ensemble_r18_mv3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.resnet = resnet18(pretrained=True)
        self.mobilenet = mobilenet_v3_large(pretrained=True)

        del self.resnet.fc

        del self.mobilenet.classifier

        self.fc = nn.Linear(2560, num_classes)

    def forward(self, x):
        x1 = self.resnet(x)
        x2 = self.mobilenet(x)

        out = torch.cat((x1, x2), dim=1)

        out = self.fc(out)

        return out


class Ensemble_d121_mv3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.densenet = densenet121(pretrained=True)
        self.mobilenet = mobilenet_v3_large(pretrained=True)

        del self.densenet.classifier

        del self.mobilenet.classifier

        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x1 = self.densenet(x)
        x2 = self.mobilenet(x)

        out = torch.cat((x1, x2), dim=1)

        out = self.fc(out)

        return out

