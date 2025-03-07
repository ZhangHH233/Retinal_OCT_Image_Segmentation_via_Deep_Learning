import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dwt_models.DWT_IDWT_layer import *
# from dwtmodel.waveletpro import Downsamplewave,Downsamplewave1

# https://github.com/yutinyang/DWAN/tree/main
# Dual Wavelet Attention Networks for Image Classification

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Downsamplewave(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsamplewave, self).__init__()
        # self.dwt = DWT_2D_tiny(wavename = wavename) #return LL
        self.dwt = DWT_2D(wavename = wavename)   #return LL,LH,HL,HH

    def forward(self, input):
        # LL = self.dwt(input)
        # return LL
        LL,LH,HL,HH = self.dwt(input)
        return torch.cat([LL,LH+HL+HH],dim=1)

class Downsamplewave1(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsamplewave1, self).__init__()
        # self.dwt = DWT_2D_tiny(wavename = wavename) #return LL
        self.dwt = DWT_2D(wavename = wavename)   #return LL,LH,HL,HH

    def forward(self, input):
        # LL = self.dwt(input)
        # return LL
        # inputori= input
        LL,LH,HL,HH = self.dwt(input)
        LL = LL+LH+HL+HH
        result = torch.sum(LL, dim=[2, 3])  # x:torch.Size([64, 256, 56, 56])
        return result  ###torch.Size([64, 256])
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        # self.att = Waveletatt(in_planes=planes)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Waveletatt(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution=224, in_planes=3, norm_layer=nn.LayerNorm):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution

        # self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.norm = norm_layer(4 * dim)
        # self.low_dim = nn.Conv2d(4 * in_planes, in_planes,kernel_size=3, stride=1,padding=1)
        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2),Downsamplewave1(wavename=wavename)])
        # self.downsamplewavelet = Downsamplewave(wavename=wavename)
        # self.conv1 = nn.Conv2d()
        # self.ac = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        xori = x
        B, C, H, W= x.shape
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1)
        # x0,x1,x2,x3 = Downsamplewave(x)
        ##x0,x1,x2,x3= self.downsamplewavelet(x)
        y = self.downsamplewavelet(x)
        y = self.fc(y).view(B, C, 1, 1)  # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        y = xori * y.expand_as(xori)       
        return y

class Waveletattspace(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution=224, in_planes=3, norm_layer=nn.LayerNorm):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution

        # self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.norm = norm_layer(4 * dim)
        # self.low_dim = nn.Conv2d(4 * in_planes, in_planes,kernel_size=3, stride=1,padding=1)
        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2),Downsamplewave(wavename=wavename)])
        # self.downsamplewavelet = Downsamplewave(wavename=wavename)
        # self.conv1 = nn.Conv2d()
        # self.ac = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.fc = nn.Sequential(
            # nn.Linear(in_planes, in_planes // 2, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Conv2d(in_planes*2, in_planes//2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes//2, in_planes,kernel_size=1,padding= 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        xori = x
        B, C, H, W= x.shape
        # H, W = self.input_resolution
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1)        
        y = self.downsamplewavelet(x)
        y = self.fc(y) # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        # y = self.fc(y).view(B, C, 1, 1)  # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        y = xori * y.expand_as(xori)       
        return y


class ResNetCIFAR(nn.Module):
    """This is a variation of ResNet for CIFAR database.
    Indeed, the network defined in Sec 4.2 performs poorly on CIFAR-100. 
    This network is similar to table 1 without the stride and max pooling
    to avoid to reduce too much the input size.
    
    This modification have been inspired by DenseNet implementation 
    for CIFAR databases.
    """
    def __init__(self, layers, num_classes=1000, levels=4):
        block = BasicBlock
        self.inplanes = 64
        super(ResNetCIFAR, self).__init__()

        self.levels = levels
        if(self.levels != 4 and self.levels != 3):
            raise "Impossible to use this number of levels"

        # Same as Densetnet
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.att1 = Waveletatt(in_planes=64)
        self.attspace1 = Waveletattspace(in_planes=64)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.att2 = Waveletatt(in_planes=128)
        self.attspace2 = Waveletattspace(in_planes=128)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        if self.levels == 4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AvgPool2d(4, stride=1)
            self.fc = nn.Linear(12800, num_classes)
        else:
            # 3 levels
            self.avgpool = nn.AvgPool2d(8, stride=1)
            self.fc = nn.Linear(256 * block.expansion, num_classes)

        self.att3 = Waveletatt(in_planes=256)
        self.att4 = Waveletatt(in_planes=512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.att2(x)
        x = self.attspace2(x)
        # x = self.att2(x)
        x = self.layer3(x)
        # x = self.att3(x)
        if self.levels == 4:
            x = self.layer4(x)
            # x = self.att4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetCIFARNormal(nn.Module):

    def __init__(self, layers, num_classes=1000):
        block = BasicBlock
        self.inplanes = 16
        super(ResNetCIFARNormal, self).__init__()
        # raise "It is not possible to use this network"
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Same as Densetnet
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



