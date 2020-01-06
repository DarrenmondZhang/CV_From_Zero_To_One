import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time


class Atrous_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):

        super(Atrous_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes,  kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class Atrous_ResNet101(nn.Module):
    def __init__(self, block, layers, pretraines=False):
        super(Atrous_ResNet101, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, rate=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, rate=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, rate=1)
        self.layer4 = self._make_mg_layer(block, 512, stride=1, rate=2)

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

    def _make_mg_layer(self, block, planes, blocks=[1, 2, 4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        conv1 = out
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # conv4 = out
        # return out, conv4
        return out, conv1


class Atrous_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(Atrous_module, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                            padding=rate, dilation=rate)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.bn(self.atrous_convolution(x))
        return x


class DeepLabV3_plus(nn.Module):
    """DeepLabV3P + ResNet101"""
    def __init__(self, num_classes, small=True, pretrained=False):
        super(DeepLabV3_plus, self).__init__()
        self.resnet_feature = Atrous_ResNet101(Atrous_Bottleneck, [3, 4, 23], pretrained)

        rates = [1, 6, 12, 18]
        self.aspp1 = Atrous_module(2048, 256, rate=rates[0])
        self.aspp2 = Atrous_module(2048, 256, rate=rates[1])
        self.aspp3 = Atrous_module(2048, 256, rate=rates[2])
        self.aspp4 = Atrous_module(2048, 256, rate=rates[3])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 256, kernel_size=1)
        )

        self.conv1x1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1),
                                     nn.BatchNorm2d(256))

        self.reduce_conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1),
                                          nn.BatchNorm2d(48))
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, x):
        # Encoder
        x, out = self.resnet_feature(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.image_pool(x)
        x5 = F.upsample(x5, size=x4.shape[2:], mode='nearest')

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1x1(x)

        # Deconder
        x = F.upsample(x, scale_factor=(4, 4), mode='bilinear')
        low_level_features = self.reduce_conv2(out)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, scale_factor=(4, 4), mode='bilinear')
        print(x.shape)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    x = torch.randn((1, 3, 256, 256))
    x = x.to(device)
    model = DeepLabV3_plus(21)
    model.cuda()
    model.eval()
    y_sppnet = model(x)
    print(y_sppnet.size())
    summary(model, (3, 256, 256))

    del y_sppnet
    torch.cuda.empty_cache()
    time.sleep(5)
