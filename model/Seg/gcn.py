import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
import time


'''
BackBone ResNet_GCN
'''


class BR(nn.Module):
    def __init__(self, out_planes):
        super(BR, self).__init__()
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1)

    def forward(self, x):
        x_res = x
        x_res = self.bn(self.relu(self.conv1(x_res)))
        x_res = self.bn(self.relu(self.conv2(x_res)))

        x = x + x_res
        return x


class Bottleneck_GCN(nn.Module):
    def __init__(self, in_plances, out_plances, kernel_size=7):
        super(Bottleneck_GCN, self).__init__()
        self.conv1_l1 = nn.Conv2d(in_channels=in_plances, out_channels=out_plances, kernel_size=(kernel_size, 1),
                                  padding=((kernel_size - 1) // 2, 0))
        self.conv1_l2 = nn.Conv2d(in_channels=out_plances, out_channels=out_plances, kernel_size=(1, kernel_size),
                                  padding=(0, (kernel_size - 1) // 2))

        self.conv2_r1 = nn.Conv2d(in_channels=in_plances, out_channels=out_plances, kernel_size=(1, kernel_size),
                                  padding=((kernel_size - 1) // 2, 0))
        self.conv2_r2 = nn.Conv2d(in_channels=out_plances, out_channels=out_plances, kernel_size=(kernel_size, 1),
                                  padding=(0, (kernel_size - 1) // 2))

        self.conv3 = nn.Conv2d(in_channels=out_plances, out_channels=in_plances, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_plances)
        # self.bn1 = nn.BatchNorm2d(in_plances)

    def forward(self, x):
        x_res_l = self.bn1(self.relu(self.conv1_l1(x)))
        x_res_l = self.bn1(self.relu(self.conv1_l2(x_res_l)))
        x_res_r = self.bn1(self.relu(self.conv2_r1(x)))
        x_res_r = self.bn1(self.relu(self.conv2_r2(x_res_r)))

        x_res = x_res_l + x_res_r
        return x_res


class FCN_GCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN_GCN, self).__init__()
        self.num_classes = num_classes
        resnet = models.resnet50(pretrained=False)
        self.conv1 = resnet.conv1  # 7x7, 64, stride=2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2  # res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3  # res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4  # res-5 o/p = 7x7,2048

        self.gcn1 = Bottleneck_GCN(256, self.num_classes)
        self.gcn2 = Bottleneck_GCN(512, self.num_classes)
        self.gcn3 = Bottleneck_GCN(1024, self.num_classes)
        self.gcn4 = Bottleneck_GCN(2048, self.num_classes)

        self.br1 = BR(self.num_classes)
        self.br2 = BR(self.num_classes)
        self.br3 = BR(self.num_classes)
        self.br4 = BR(self.num_classes)
        self.br5 = BR(self.num_classes)
        self.br6 = BR(self.num_classes)
        self.br7 = BR(self.num_classes)
        self.br8 = BR(self.num_classes)
        self.br9 = BR(self.num_classes)

    def forward(self, x):
        origin_input = x
        print(origin_input.size)
        input = self.relu(self.bn1(self.conv1(x)))

        x = self.maxpool(input)

        res_2 = self.layer1(x)
        res_3 = self.layer2(res_2)
        res_4 = self.layer3(res_3)
        res_5 = self.layer4(res_4)

        gcn_1 = self.br1(self.gcn1(res_2))
        gcn_2 = self.br2(self.gcn2(res_3))
        gcn_3 = self.br3(self.gcn3(res_4))
        gcn_4 = self.br4(self.gcn4(res_5))

        gcn_4_up = F.upsample(input=gcn_4, size=res_4.size()[2:], mode='bilinear', align_corners=True)
        gcn_3_up = F.upsample(input=self.br5(gcn_4_up+gcn_3), size=res_3.size()[2:], mode='bilinear', align_corners=True)
        gcn_2_up = F.upsample(input=self.br6(gcn_3_up+gcn_2), size=res_2.size()[2:], mode='bilinear', align_corners=True)
        gcn_1_up = F.upsample(input=self.br7(gcn_2_up+gcn_1), size=input.size()[2:], mode='bilinear', align_corners=True)

        gcn_0_up = F.upsample(input=self.br8(gcn_1_up), scale_factor=2, mode='bilinear', align_corners=True)
        out = F.upsample(input=self.br8(gcn_0_up), size=origin_input.size()[2:], mode='bilinear', align_corners=True)
        print(out.size)
        return out

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    x = torch.randn((1, 3, 256, 256))
    x = x.to(device)
    model = FCN_GCN(21)
    model.cuda()
    model.eval()
    y_sppnet = model(x)
    print(y_sppnet.size())
    summary(model, (3, 256, 256))

    del y_sppnet
    torch.cuda.empty_cache()
    time.sleep(5)