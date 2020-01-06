import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding and no bias"""
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    """3x3 convolution with padding bn relu and no bias"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True) )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution and no bias; downsample 1/stride"""
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False)


def atrous_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3  atrous convolution and no bias"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = atrous_conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class PSPBlock(nn.Module):
    """Pyramid Pooling Module"""
    def __init__(self, levels, in_planes, out_planes=512):
        super(PSPBlock, self).__init__()
        self.levels = levels
        self.convblock = nn.Sequential(
            conv1x1(in_planes=in_planes, out_planes=out_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        x = F.adaptive_avg_pool2d(input=x, output_size=self.levels)
        x = self.convblock(x)
        x = F.upsample(input=x, size=size, mode='bilinear', align_corners=True)
        return x


class SPP(nn.Module):
    def __init__(self, in_channel=2048):
        super().__init__()
        self.spp1 = PSPBlock(levels=1, in_planes=in_channel)
        self.spp2 = PSPBlock(levels=2, in_planes=in_channel)
        self.spp3 = PSPBlock(levels=3, in_planes=in_channel)
        self.spp6 = PSPBlock(levels=6, in_planes=in_channel)

    def forward(self, x):
        # x 2048 num_output
        x1 = self.spp1(x)
        x2 = self.spp2(x)
        x3 = self.spp3(x)
        x6 = self.spp6(x)
        out = torch.cat([x, x1, x2, x3, x6], dim=1)

        return out


class SPPNet(nn.Module):
    def __init__(self, block, layers, class_num, dropout_rate=0.2, groups=1,
                 width_per_group=64, norm_layer=None, replace_stride_with_dilation=None):
        super(SPPNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        self.spp = SPP(in_channel=2048)
        self.conv5_4 = conv3x3_bn_relu(2048 + 512 * 4, 512)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv6 = nn.Conv2d(512, class_num, 1, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        size = x.shape[2:]
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.spp(x)
        x = self.conv5_4(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = F.upsample(x, size, mode='bilinear', align_corners=True)

        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    x = torch.randn((1, 3, 256, 256))
    x = x.to(device)
    model = SPPNet(Bottleneck, layers=[3, 4, 23, 3], class_num=21, dropout_rate=0.5)
    model.cuda()
    model.eval()
    y_sppnet = model(x)
    print(y_sppnet.size())
    summary(model, (3, 256, 256))

    del y_sppnet
    torch.cuda.empty_cache()
    time.sleep(5)
