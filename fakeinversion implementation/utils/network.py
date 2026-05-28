import torch, torchvision
import torch.nn as nn
from typing import Literal

#returns modified resnet model
#should use insrance norm according to author fakinversion author
def get_resnet_model( residuals: bool, norm_layer: Literal["batch", "instance"], pretrained: bool) -> nn.Module:

    if norm_layer == "batch":
        norm = nn.BatchNorm2d
    elif norm_layer == "instance":
        norm = nn.InstanceNorm2d
    else:
        raise AssertionError("Unknown norm layer")

    model = torchvision.models.resnet50(num_classes=1000, pretrained=pretrained, norm_layer=norm)

    if residuals:
        model.conv1.weight = nn.Parameter(torch.cat([model.conv1.weight * 0.25] * 4, dim=1))
        model.conv1.in_channels = 12 # 4 x rgb when including residuals
    else:
        model.conv1.weight = nn.Parameter(torch.cat([model.conv1.weight * (1/3)] * 3, dim=1))
        model.conv1.in_channels = 8

    model.fc = nn.Linear(2048, 1)
    #populating new fully conected layer with small weight for more stable finetuning of resnet network
    torch.nn.init.normal_(model.fc.weight.data, 0.0, 0.02) 

    return model