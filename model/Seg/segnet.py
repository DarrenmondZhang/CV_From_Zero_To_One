
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time


class SegNet_BN_ReLU(nn.Module):
    """
    SegNet-VGG
    """

    def __init__(self, in_planes, out_planes):
        super(SegNet_BN_ReLU, self).__init__()

        self.in_plance = in_planes
        self.out_plance = out_planes
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        # encoder
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
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)

        # decoder
        self.conv5_1_up = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_1_bn_up = nn.BatchNorm2d(512)
        self.conv5_2_up = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2_bn_up = nn.BatchNorm2d(512)
        self.conv5_3_up = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3_bn_up = nn.BatchNorm2d(512)

        self.conv4_1_up = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_1_bn_up = nn.BatchNorm2d(512)
        self.conv4_2_up = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_bn_up = nn.BatchNorm2d(512)
        self.conv4_3_up = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv4_3_bn_up = nn.BatchNorm2d(256)

        self.conv3_1_up = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_1_bn_up = nn.BatchNorm2d(256)
        self.conv3_2_up = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2_bn_up = nn.BatchNorm2d(256)
        self.conv3_3_up = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv3_3_bn_up = nn.BatchNorm2d(128)

        self.conv2_1_up = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv2_1_bn_up = nn.BatchNorm2d(128)
        self.conv2_2_up = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv2_2_bn_up = nn.BatchNorm2d(64)

        self.conv1_1_up = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_1_bn_up = nn.BatchNorm2d(64)
        self.conv1_2_up = nn.Conv2d(in_channels=64, out_channels=out_planes, kernel_size=3, padding=1)
        self.conv1_2_bn_up = nn.BatchNorm2d(out_planes)

        # self.apply(self.weight_init)

    def forward(self, x):
        # Encoder block1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        size1 = x.size()
        x, max_index1 = self.pool(x)

        # Encoder block2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        size2 = x.size()
        x, max_index2 = self.pool(x)

        # Encoder block3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        size3 = x.size()
        x, max_index3 = self.pool(x)

        # Encoder block4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        size4 = x.size()
        x, max_index4 = self.pool(x)

        # Encoder block5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        size5 = x.size()
        x, max_index5 = self.pool(x)

        # Decoder block5
        x = self.unpool(x, max_index5, output_size=size5)
        x = self.conv5_1_bn_up(F.relu(self.conv5_1_up(x)))
        x = self.conv5_2_bn_up(F.relu(self.conv5_2_up(x)))
        x = self.conv5_3_bn_up(F.relu(self.conv5_3_up(x)))

        # Decoder block4
        x = self.unpool(x, max_index4, output_size=size4)
        x = self.conv4_1_bn_up(F.relu(self.conv4_1_up(x)))
        x = self.conv4_2_bn_up(F.relu(self.conv4_2_up(x)))
        x = self.conv4_3_bn_up(F.relu(self.conv4_3_up(x)))

        # Decoder block3
        x = self.unpool(x, max_index3, output_size=size3)
        x = self.conv3_1_bn_up(F.relu(self.conv3_1_up(x)))
        x = self.conv3_2_bn_up(F.relu(self.conv3_2_up(x)))
        x = self.conv3_3_bn_up(F.relu(self.conv3_3_up(x)))

        # Decoder block2
        x = self.unpool(x, max_index2, output_size=size2)
        x = self.conv2_1_bn_up(F.relu(self.conv2_1_up(x)))
        x = self.conv2_2_bn_up(F.relu(self.conv2_2_up(x)))

        # Decoder block1
        x = self.unpool(x, max_index1, output_size=size1)
        x = self.conv1_1_bn_up(F.relu(self.conv1_1_up(x)))
        x = self.conv1_2_bn_up(F.relu(self.conv1_2_up(x)))

        return x


def segnet_bn_relu(in_planes, out_planes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SegNet_BN_ReLU(in_planes, out_planes)
    if pretrained:
        model.load_pretrained_weights()
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
x = torch.randn((1, 3, 256, 256))
x = x.to(device)
model = segnet_bn_relu(3, 21)
# model = VGG_19bn_8s(21)
model.cuda()
model.eval()
y_Seg = model(x)
print(y_Seg.size())
summary(model, (3, 256, 256))

del y_Seg
torch.cuda.empty_cache()
time.sleep(5)