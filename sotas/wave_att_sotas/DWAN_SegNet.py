import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
items = current_dir.split('/')
for item in items:
    if 'v2024' in item or  'V2024' in item:
        obj_name = item
obj_name = os.path.abspath('./DWT_2023/{}'.format(obj_name)) # add current object name to sys
sys.path.append(obj_name)

from models.blocks.dwt_modules.DWT_IDWT_layer import *
# from models.dwt_models.DWT_IDWT_layer import *
# from dwtmodel.waveletpro import Downsamplewave,Downsamplewave1

# https://github.com/yutinyang/DWAN/tree/main
# Dual Wavelet Attention Networks for Image Classification
def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return down_conv

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
        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2),Downsamplewave(wavename=wavename)])
       
        self.fc = nn.Sequential(           
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
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1)        
        y = self.downsamplewavelet(x)
        y = self.fc(y) # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        # y = self.fc(y).view(B, C, 1, 1)  # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        y = xori * y.expand_as(xori)       
        return y

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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class DWAN_SegNet(nn.Module):
    def __init__(self,in_channels = 3, num_classes =4,filters=[64, 128, 256, 512]):
        super(DWAN_SegNet,self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )        
        
        self.enc1 = nn.Sequential(BasicBlock(filters[0], filters[0]),
                                  nn.Conv2d(filters[0],filters[1],1)
                                  )                                  
                                  
        self.enc2 = nn.Sequential(BasicBlock(filters[1], filters[1]),
                                  nn.Conv2d(filters[1],filters[2],1)
                                  ) 
                                  
        self.enc3 = nn.Sequential(BasicBlock(filters[2], filters[2]),
                                  nn.Conv2d(filters[2],filters[3],1)
                                  )
        
        self.middle = x2conv(filters[3],filters[3])  
        
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], 2, 2)
        self.dec3 = nn.Sequential(BasicBlock(filters[3], filters[3]),
                                  nn.Conv2d(filters[3],filters[2],1)
                                  )
                                  
        
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], 2, 2)
        self.dec2 = nn.Sequential(BasicBlock(filters[2], filters[2]),
                                  nn.Conv2d(filters[2],filters[1],1)
                                  )
        
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], 2, 2)
        self.dec1 = nn.Sequential(BasicBlock(filters[1], filters[1]),
                                  nn.Conv2d(filters[1],filters[0],1)
                                  )
        
        self.output_layer = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.enc1(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.enc2(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.enc3(x4)
        
        x = self.middle(x4)
        
        d3 = self.up3(x)
        d3 = torch.cat((x3,d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((x2,d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((x1,d1), dim=1)
        d1 = self.dec1(d1)
        
        output = self.output_layer(d1)
        return output

if __name__=='__main__': 
    import os
    import sys
    CUDA_LAUNCH_BLOCKING=1
    os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
    
    

    # from models.sota_models.MWG_Net.VGGNET import VGG

    model =DWAN_SegNet(in_channels=3, num_classes=4).cuda()
    
    input = torch.randn(1,3,512,512)
    
    input = input.cuda()
    
    out = model(input)
    
    print(out)