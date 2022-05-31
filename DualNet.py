import torch
import torch.nn as nn
import torch.nn.functional as F
from srm_1order import *

class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,strides=None,padding=0,ceil_mode = False,count_include_pad = True,divisor_override = None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=kernel_size,stride=strides,padding=padding)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SoftPooling2D(kernel_size=(2,2),strides=(2,2),padding=0)
        )

    def forward(self, x):
        out = self.process(x)
        return out

class One_Conv(nn.Module):
    def __init__(self, in_channels):
        super(One_Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        return x

class Other_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Other_Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        return x

class Pre(nn.Module):
    def __init__(self):
        super(Pre, self).__init__()
        hpf_list = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            hpf_list.append(hpf_item)
        # list1 = np.concatenate((all_hpf_list_5, all_hpf_list_5), axis=-2)
        # hpf_list = np.concatenate((list1, all_hpf_list_5), axis=-2)
        hpf_weight = nn.Parameter(torch.Tensor(hpf_list).view(8, 1, 5, 5), requires_grad=False)
        self.hpf = nn.Conv2d(1, 8, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

    def forward(self, x):
        pre = self.hpf(x)
        return pre

class NoiseModule(nn.Module):
    def __init__(self):
        super(NoiseModule, self).__init__()
        # self.hpf = nn.Conv2d(1, 3, 5, 1, 2)
        # self.hpf.weight = nn.Parameter(torch.tensor(filter_3).view(3, 1, 5, 5), requires_grad=False)
        # self.hsv = RGB_HSV()
        self.pre1 = Pre()
        self.pre2 = Pre()
        self.pre3 = Pre()
        self.block0 = Block(24, 128)
        #self.conv0 = Other_Conv(90,128)
        self.block1 = Block(128, 128)
        self.conv1 = One_Conv(128)
        self.block2 = Block(128, 128)
        self.conv2 = One_Conv(128)
        self.block3 = Block(128, 128)
        self.conv3 = One_Conv(128)
        self.block4 = Block(128, 128)
        #self.conv4 = One_Conv(128)
        # self.resnet4 = ResNet18(7, 3)
        # 28*28
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128, 2)
        # self.fc2 = nn.Linear(1028, 2)
        #self.softmax = nn.Softmax(dim=1)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, inp):
        # img = inp.index_select(1, torch.tensor([0, 1, 2]).cuda())
        # gray = inp.index_select(1, torch.tensor([3]).cuda())
        # res = self.hpf(gray)
        # inp =  self.hsv.rgb_to_hsv(inp)
        pre1 = self.pre1(inp[:, 0, :, :].view(inp.size(0), 1, 224, 224))
        pre2 = self.pre2(inp[:, 1, :, :].view(inp.size(0), 1, 224, 224))
        pre3 = self.pre3(inp[:, 2, :, :].view(inp.size(0), 1, 224, 224))
        pre = torch.cat((pre1, pre2, pre3), dim=1)

        x = self.block0(pre)
        #x1 = self.conv0(pre)
        #x = x0 + x1
        #x = self.relu(x)

        x0 = self.block1(x)
        x1 = self.conv1(x)
        x = x0 + x1
        x = self.relu(x)

        x0 = self.block2(x)
        x1 = self.conv2(x)
        x = x0 + x1
        x = self.relu(x)

        x0 = self.block3(x)
        x1 = self.conv3(x)
        x = x0 + x1
        x = self.relu(x)

        x = self.block4(x)
        #x1 = self.conv4(x)
        #x = x0 + x1
        #x = self.relu(x)

        output = F.adaptive_avg_pool2d(x, (1, 1))
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        #out = self.softmax(out)
        return output

class RgbModule(nn.Module):
    def __init__(self):
        super(RgbModule, self).__init__()
        # self.hpf = nn.Conv2d(1, 3, 5, 1, 2)
        # self.hpf.weight = nn.Parameter(torch.tensor(filter_3).view(3, 1, 5, 5), requires_grad=False)
        # self.hsv = RGB_HSV()
        self.block0 = Block(3, 32)
        self.block1 = Block(32, 64)
        self.block2 = Block(64, 128)
        self.block3 = Block(128, 128)
        self.block4 = Block(128, 128)
        #self.conv4 = Other_Conv(128,128)
        self.relu = nn.ReLU(inplace=True)
        # self.resnet4 = ResNet18(7, 3)
        # 28*28
        self.fc1 = nn.Linear(128, 2)
        # self.fc2 = nn.Linear(1028, 2)
        #self.softmax = nn.Softmax(dim=1)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, rgb):
        # img = inp.index_select(1, torch.tensor([0, 1, 2]).cuda())
        # gray = inp.index_select(1, torch.tensor([3]).cuda())
        # res = self.hpf(gray)
        # inp =  self.hsv.rgb_to_hsv(inp)
        x = self.block0(rgb)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

class DualNet(nn.Module):
    def __init__(self):
        super(DualNet, self).__init__()
        self.noisemodule = NoiseModule()
        self.rgbmodule = RgbModule()

    def forward(self, rgb):
        noise = self.noisemodule(rgb)
        rgb = self.rgbmodule(rgb)

        return noise+rgb



