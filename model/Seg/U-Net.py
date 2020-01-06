import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time


class UNet(nn.Module):
    """
    U-Net
    """
    def __init__(self, in_planes, out_planes):
        super(UNet, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv1_1 = nn.Conv2d(in_channels=in_planes, out_channels=64, kernel_size=3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(1024)
        self.conv5_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, stride=2,
                                          output_padding=1)
        self.conv5_1_up = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.conv5_1_bn_up = nn.BatchNorm2d(512)
        self.conv5_2_up = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2_bn_up = nn.BatchNorm2d(512)

        self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2,
                                          output_padding=1)

        self.conv4_1_up = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv4_1_bn_up = nn.BatchNorm2d(256)
        self.conv4_2_up = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv4_2_bn_up = nn.BatchNorm2d(256)

        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2,
                                          output_padding=1)
        self.conv3_1_up = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv3_1_bn_up = nn.BatchNorm2d(128)
        self.conv3_2_up = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv3_2_bn_up = nn.BatchNorm2d(128)

        self.upconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2,
                                          output_padding=1)
        self.conv2_1_up = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv2_1_bn_up = nn.BatchNorm2d(64)
        self.conv2_2_up = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_2_bn_up = nn.BatchNorm2d(64)

        self.conv1_1_up = nn.Conv2d(in_channels=64, out_channels=out_planes, kernel_size=1)
        self.conv1_1_bn_up = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x1 = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x1 = self.conv1_2_bn(F.relu(self.conv1_2(x1)))
        pool1 = self.pool(x1)

        x2 = self.conv2_1_bn(F.relu(self.conv2_1(pool1)))
        x2 = self.conv2_2_bn(F.relu(self.conv2_2(x2)))
        pool2 = self.pool(x2)

        x3 = self.conv3_1_bn(F.relu(self.conv3_1(pool2)))
        x3 = self.conv3_2_bn(F.relu(self.conv3_2(x3)))
        pool3 = self.pool(x3)

        x4 = self.conv4_1_bn(F.relu(self.conv4_1(pool3)))
        x4 = self.conv4_2_bn(F.relu(self.conv4_2(x4)))
        pool4 = self.pool(x4)

        x5 = self.conv5_1_bn(F.relu(self.conv5_1(pool4)))
        x5 = self.conv5_2_bn(F.relu(self.conv5_2(x5)))

        # Decoder + skip connection
        x52 = self.upconv4(x5)
        x5_up = torch.cat((self.upconv4(x5), x4), 1)
        print(x5_up.shape)
        x5_up = self.conv5_1_bn_up(F.relu(self.conv5_1_up(x5_up)))
        x5_up = self.conv5_2_bn_up(F.relu(self.conv5_2_up(x5_up)))

        x4_up = torch.cat((self.upconv3(x4), x3), 1)
        print(x4_up.shape)
        x4_up = self.conv4_1_bn_up(F.relu(self.conv4_1_up(x4_up)))
        x4_up = self.conv4_2_bn_up(F.relu(self.conv4_2_up(x4_up)))

        x3_up = torch.cat((self.upconv2(x3), x2), 1)
        print(x3_up.shape)
        x3_up = self.conv3_1_bn_up(F.relu(self.conv3_1_up(x3_up)))
        x3_up = self.conv3_2_bn_up(F.relu(self.conv3_2_up(x3_up)))

        x2_up = torch.cat((self.upconv1(x2), x1), 1)
        print(x2_up.shape)
        x2_up = self.conv2_1_bn_up(F.relu(self.conv2_1_up(x2_up)))
        x2_up = self.conv2_2_bn_up(F.relu(self.conv2_2_up(x2_up)))

        x1_up = self.conv1_1_bn_up(F.relu((self.conv1_1_up(x2_up))))
        return x1_up


def unet_bn_relu(in_planes, out_planes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = UNet(in_planes, out_planes)
    if pretrained:
        model.load_pretrained_weights()
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
x = torch.randn((1, 3, 256, 256))
x = x.to(device)
model = unet_bn_relu(3, 21)
model.cuda()
model.eval()
y_unet = model(x)
print(y_unet.size())
summary(model, (3, 256, 256))

del y_unet
torch.cuda.empty_cache()
time.sleep(5)

