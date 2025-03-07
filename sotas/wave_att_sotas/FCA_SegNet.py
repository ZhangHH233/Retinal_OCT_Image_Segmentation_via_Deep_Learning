import math
import torch

import torch.nn as nn

# from DWT_2023.v20240304.models.sota_models.WA_Nets.FCA_Net import FcaBasicBlock


class Fca_SegNet(nn.Module):
    def __init__(self,in_channels = 3, num_classes =4,filters=[64, 128, 256, 512]):
        super(Fca_SegNet,self).__init__()
        
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
        
        self.enc1 = nn.Sequential(FcaBasicBlock(filters[0], filters[0]),
                                  nn.Conv2d(filters[0],filters[1],1)
                                  )                                  
                                  
        self.enc2 = nn.Sequential(FcaBasicBlock(filters[1], filters[1]),
                                  nn.Conv2d(filters[1],filters[2],1)
                                  ) 
                                  
        self.enc3 = nn.Sequential(FcaBasicBlock(filters[2], filters[2]),
                                  nn.Conv2d(filters[2],filters[3],1)
                                  )
        
        self.middle = x2conv(filters[3],filters[3])  
        
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], 2, 2)
        self.dec3 = nn.Sequential(FcaBasicBlock(filters[3], filters[3]),
                                  nn.Conv2d(filters[3],filters[2],1)
                                  )
                                  
        
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], 2, 2)
        self.dec2 = nn.Sequential(FcaBasicBlock(filters[2], filters[2]),
                                  nn.Conv2d(filters[2],filters[1],1)
                                  )
        
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], 2, 2)
        self.dec1 = nn.Sequential(FcaBasicBlock(filters[1], filters[1]),
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

class FcaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16, ):
        global _mapper_x, _mapper_y
        super(FcaBasicBlock, self).__init__()
        # assert fea_h is not None
        # assert fea_w is not None
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
        self.downsample = downsample
        self.stride = stride
        
        

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
       

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter
    
    

if __name__=='__main__': 
    import os
    import sys
    CUDA_LAUNCH_BLOCKING=1
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'

    model = FCA_SegNet(in_channels=3, num_classes=4).cuda()
    
    input = torch.randn(4,3,512,512)
    
    input = input.cuda()
    
    out = model(input)
    
    print(out)