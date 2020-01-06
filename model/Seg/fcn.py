
import math
import torch
import torch.nn as nn
import torch.nn.init as init


class Block(nn.Module):
    """
    基础卷积块
    """

    def __init__(self, in_planes, out_planes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv1(x)))
        return out


def _make_layers(in_planes, layer_list):
    layers = []
    for i in layer_list:
        layers.append(Block(in_planes, i))
        in_planes = i  # update the next input_channels
    return nn.Sequential(*layers)


class Layers(nn.Module):
    def __init__(self, in_planes, layer_list):
        super(Layers, self).__init__()
        self.layer = _make_layers(in_planes, layer_list)

    def forward(self, x):
        out = self.layer(x)
        return out


class VGG(nn.Module):
    """
    建立VGG-19BN--encode模型
    """

    def __init__(self):
        super(VGG, self).__init__()
        self.layer1 = Layers(3, [64, 64])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = Layers(64, [128, 128])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = Layers(128, [256, 256, 256, 256])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = Layers(256, [512, 512, 512, 512])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = Layers(512, [512, 512, 512, 512])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        f1 = self.pool1(self.layer1(x))
        f2 = self.pool2(self.layer2(f1))
        f3 = self.pool3(self.layer3(f2))
        f4 = self.pool4(self.layer4(f3))
        f5 = self.pool5(self.layer5(f4))

        return [f3, f4, f5]


class FCNDecode(nn.Module):
    """
    建立上采样模块
    """

    def __init__(self, n, in_channels, out_channels, upsample_ratio):
        super(FCNDecode, self).__init__()
        self.conv1 = Layers(in_channels, [out_channels] * 4)
        self.trans_conv1 = nn.ConvTranspose2d(
            out_channels,
            out_channels,
            upsample_ratio,
            stride=upsample_ratio
        )

    def forward(self, x):
        out = self.trans_conv1(self.conv1(x))
        return out


class MergeUpsample(nn.Module):
    def __init__(self, in_planes1, in_planes2, out_planes, upsample_ratio):
        super(MergeUpsample, self).__init__()
        self.conv1 = Block(in_planes1, out_planes)
        self.conv2 = Block(in_planes2, out_planes)
        self.conv3 = Block(out_planes, out_planes)
        self.upsample1 = nn.ConvTranspose2d(
            out_planes,
            out_planes,
            upsample_ratio,
            stride=upsample_ratio
        )
        self.add = nn.AdaptiveAvgPool2d

    def forward(self, x, y):
        p1 = self.conv1(self.upsample1(x))
        p2 = self.conv2(y)
        out = self.conv2(p1+p2)
        return out


class FCNSeg(nn.Module):
    def __init__(self, n, in_planes, out_planes, upsample_ratio):
        super(FCNSeg, self).__init__()
        self.encode = VGG()
        self.decode = FCNDecode(n, in_planes, out_planes, upsample_ratio)
        self.classifier = nn.Conv2d(out_planes, 10, 3, padding=1)

    def forward(self, x):
        feature_list = self.encode(x)
        out = self.decode(feature_list[-1])
        result = self.classifier(out)

        return result


x = torch.randn((10, 3, 256, 256))
model = FCNSeg(4, 512, 256, 32)
model.eval()
y = model(x)
print(y.size())
