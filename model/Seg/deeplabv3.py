import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time


class Atrous_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Atrous_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Atrous_ResNet(nn.Module):
    def __init__(self, block, layers, pretrained=False):
        super(Atrous_ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, rate=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, rate=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, rate=1)
        self.layer4 = self._make_mg_unit(block, 512, stride=1, rate=2)

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_mg_unit(self, block, planes, blocks=[1, 2, 4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0] * rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i] * rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class Atrous_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(Atrous_module, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                            padding=rate, dilation=rate)
        self.batch_norm = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.batch_norm(self.atrous_convolution(x)))

        return x


class DeepLabv3(nn.Module):
    """DeepLabV3 + ResNet101"""
    def __init__(self, num_classes, pretrained=False):
        super(DeepLabv3, self).__init__()
        self.resnet_features = Atrous_ResNet(Atrous_Bottleneck, [3, 4, 23], pretrained)

        rates = [1, 6, 12, 18]
        self.aspp1 = Atrous_module(2048, 256, rate=rates[0])
        self.aspp2 = Atrous_module(2048, 256, rate=rates[1])
        self.aspp3 = Atrous_module(2048, 256, rate=rates[2])
        self.aspp4 = Atrous_module(2048, 256, rate=rates[3])

        self.image_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(2048, 256, kernel_size=1)
        )
        self.fc1 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        self.fc2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.resnet_features(x)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = self.image_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='nearest')

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.upsample(x, scale_factor=(16, 16), mode='bilinear')

        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    x = torch.randn((1, 3, 256, 256))
    x = x.to(device)
    model = DeepLabv3(21)
    model.cuda()
    model.eval()
    y_sppnet = model(x)
    print(y_sppnet.size())
    summary(model, (3, 256, 256))

    del y_sppnet
    torch.cuda.empty_cache()
    time.sleep(5)
