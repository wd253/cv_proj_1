import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))


class CenterNet(nn.Module):
    def __init__(self, num_classes=4, pretrained_backbone=True):
        super(CenterNet, self).__init__()
        self.num_classes = num_classes

        # Backbone (ResNet-18)
        resnet = resnet18(pretrained=pretrained_backbone)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # Deconvolution layers (upsample x2 each)
        self.deconv_layers = nn.Sequential(
            DeconvBlock(512, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64)
        )

        # Output heads
        self.hm_head = nn.Conv2d(64, num_classes, kernel_size=1)        # heatmap head
        self.wh_head = nn.Conv2d(64, 2, kernel_size=1)                  # width-height head (optional)
        self.offset_head = nn.Conv2d(64, 2, kernel_size=1)              # offset head (optional)

        self._init_head(self.hm_head)
        self._init_head(self.wh_head)
        self._init_head(self.offset_head)

    def _init_head(self, head):
        for m in head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.deconv_layers(feat)
        hm = torch.sigmoid(self.hm_head(feat))
        wh = self.wh_head(feat)
        offset = self.offset_head(feat)
        return hm, wh, offset
