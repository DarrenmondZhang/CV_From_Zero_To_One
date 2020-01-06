import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                     bias=False)


def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layers=None, downsample=None):
        super(BasicBlock, self).__init__()
        if norm_layers is None:
            norm_layers = nn.BatchNorm2d()  # 如果没有自定义BN，则使用系统自带的
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layers(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = norm_layers(planes)
        self.downsample = downsample  # 调整feature map维度
        self.stride = stride

    def forward(self, x):
        identity = x  # save shortcut，保存输入，和FCN类似

        out = self.conv1(x)
        out = self.bn1(out)  # batch_normalization
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu1(out)

        return out


class Bottleneck(nn.Module):

    def __init__(self, in_planes, planes, stride=1, norm_layers=None, downsample=None):
        super(Bottleneck, self).__init__()
        if norm_layers is None:
            norm_layers = nn.BatchNorm2d()

        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = norm_layers(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layers(planes)
        self.conv3 = conv1x1(planes, in_planes)
        self.bn3 = norm_layers(in_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2)
        out = self.relu(self.bn3(self.conv3(x)))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_class=1000, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d()
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # layers:[3, 4, 23, 3]  -->  ResNet101
        # layers:[3, 4, 6, 3]  -->  ResNet34
        # model = ResNet(..., layers)

        self.GAP = nn.AdaptiveAvgPool2d((1, 1))  # （1,1) <==> GAP
        self.fc = nn.Linear(512, num_class)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, norm_layer))
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)  # 把模块拼接成一个Sequential层，调用函数直接加入到网络中

    def forward(self, x):

        ###### part-one ##########
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.pool1(out)

        ###### part-two to five ##########
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.GAP(out)
        out = torch.flatten(out, 1)  # use one fc layer
        out = self.fc(out)

        return out


def resnet34(pretrained=False):
    return ResNet('resnet34', BasicBlock, [3, 4, 6, 3])


def resnet50(pretrained=False):
    return ResNet('resnet34', Bottleneck, [3, 4, 6, 3])


def resnet101(pretrained=False):
    return ResNet('resnet34', Bottleneck, [3, 4, 23, 3])