import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time
from model.BackBone.Xception import Xception


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
        self.resnet_feature = Xception()

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
        x, low_level_feat = self.resnet_feature(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.image_pool(x)
        x5 = F.upsample(x5, size=x4.shape[2:], mode='nearest')

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1x1(x)

        # Deconder
        x = F.upsample(x, scale_factor=(2, 2), mode='bilinear')
        low_level_feat = self.reduce_conv2(low_level_feat)

        x = torch.cat((x, low_level_feat), dim=1)
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
