import torch
from torch import nn


class VGG16(nn.Module):

    def __init__(self, n_classes=21):
        super(VGG16, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)  # samepadding
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 尺寸 * 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # samepadding
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 尺寸 * 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # samepadding
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # samepadding
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 尺寸 * 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # samepadding
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # samepadding
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 尺寸 * 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # samepadding
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # samepadding
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 尺寸 * 1/32

        # self.fc6 = nn.Linear(512, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        # self.fc8 = nn.Linear(4096, n_classes)

        # fc6
        self.fc6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7)  # full-sized kernel
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout2d()
        # fc7
        self.fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)  # full-sized kernel
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout2d()
        # coarse output
        self.score = nn.Conv2d(in_channels=4096, out_channels=n_classes, kernel_size=1)  # full-sized kernel

        # FCN-32s
        # self.upscore = nn.ConvTranspose2d(n_classes, n_classes, 64, stride=32)

        # FCN-16s
        self.upscore2 = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2)  # *2
        self.upscore16 = nn.ConvTranspose2d(n_classes, n_classes, 32, stride=16)  # *16

        # FCN-8s
        self_upscore8 = nn.ConvTranspose2d(n_classes, n_classes, 16, stride=8)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.dropout6(h)

        h = self.relu7(self.fc7(h))
        h = self.dropout7(h)

        h = self.score(h)
        # FCN-32s
        # h = self.upscore(h)

        # FCN-16s
        upscore2 = self.upscore2(h)  # *2
        score_pool4 = pool4
        h = score_pool4 + upscore2  # spatial is same! but channels is same???
        h = self.upscore16(h)  # *16

        # FCN-8s
        h = h + pool3
        h = self.upscore8(h)

        return h