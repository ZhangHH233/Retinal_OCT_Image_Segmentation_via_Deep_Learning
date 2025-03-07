import torch
from torch import nn
import torch.nn.functional as F

from models.sotas.wave_att_sotas.VGGNET import VGG

import pywt
import pywt.data

# https://github.com/MLMIP/MWG-Net/tree/main
# MWG-Net: Multi-scale Wavelet Guidance Network for COVID-19 Lung Infection Segmentation from CT Images
"""
@article{Hu_2023_MWG, 
title={MWG-Net: Multi-scale Wavelet Guidance Network for COVID-19 Lung Infection Segmentation from CT Images},  
url={http://dx.doi.org/10.1109/tim.2023.3265100},  
DOI={10.1109/tim.2023.3265100},  
journal={IEEE Transactions on Instrumentation and Measurement}, 
author={Hu, Kai and Tan, Hui and others},  
year={2023}, 
month={Jan},  
pages={1–15}, 
language={en-US}  }
"""
def dwt_init(x):
     coeffs2 = pywt.dwt2(x, 'haar')
     x_LL, (x_LH, x_HL, x_HH) = coeffs2
     return x_LL, x_HL, x_LH, x_HH


# from models.dwt_models.DWT_IDWT_layer import DWT_2D as dwt_init

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


class DWT_1(nn.Module):
    def __init__(self, J=1):
        super(DWT_1, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x_LL_1, x_HL_1, x_LH_1, x_HH_1 = dwt_init(x)
        return x_LL_1, x_HL_1, x_LH_1, x_HH_1


class DWT_2(nn.Module):
    def __init__(self, J=2):
        super(DWT_2, self).__init__()
        self.requires_grad = False
        self.DWT_1 = DWT_1()

    def forward(self, x):
        x_LL_1, x_HL_1, x_LH_1, x_HH_1 = self.DWT_1(x)
        x_LL_2, x_HL_2, x_LH_2, x_HH_2 = dwt_init(x_LL_1)
        return x_LL_2, x_HL_2, x_LH_2, x_HH_2


class DWT_3(nn.Module):
    def __init__(self, J=3):
        super(DWT_3, self).__init__()
        self.requires_grad = False
        self.DWT_2 = DWT_2()

    def forward(self, x):
        x_LL_2, x_HL_2, x_LH_2, x_HH_2 = self.DWT_2(x)
        x_LL_3, x_HL_3, x_LH_3, x_HH_3 = dwt_init(x_LL_2)
        return x_LL_3, x_HL_3, x_LH_3, x_HH_3


class DWT_4(nn.Module):
    def __init__(self, J=4):
        super(DWT_4, self).__init__()
        self.requires_grad = False
        self.DWT_3 = DWT_3()

    def forward(self, x):
        x_LL_3, x_HL_3, x_LH_3, x_HH_3 = self.DWT_3(x)
        x_LL_4, x_HL_4, x_LH_4, x_HH_4 = dwt_init(x_LL_3)
        return x_LL_4, x_HL_4, x_LH_4, x_HH_4


class BasicConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv_sigmoid(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), )

    def forward(self, input):
        return self.conv(input)


class DWT_block_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWT_block_1, self).__init__()

        self.DWT = DWT_1()
        self.Conv = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1), )

    def forward(self, x):
        x_LL, x_HL, x_LH, x_HH = self.DWT(x)
        cat = torch.cat([x_LL, x_HL, x_LH], dim=1)
        out = self.Conv(cat)

        return out


class DWT_block_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWT_block_2, self).__init__()

        self.DWT = DWT_2()
        self.Conv = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1), )

    def forward(self, x):
        x_LL, x_HL, x_LH, x_HH = self.DWT(x)
        cat = torch.cat([x_LL, x_HL, x_LH], dim=1)
        out = self.Conv(cat)

        return out


class DWT_block_3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWT_block_3, self).__init__()

        self.DWT = DWT_3()
        self.Conv = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1), )

    def forward(self, x):
        x_LL, x_HL, x_LH, x_HH = self.DWT(x)
        cat = torch.cat([x_LL, x_HL, x_LH], dim=1)
        out = self.Conv(cat)

        return out


class DWT_block_4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWT_block_4, self).__init__()

        self.DWT = DWT_4()
        self.Conv = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1), )

    def forward(self, x):
        x_LL, x_HL, x_LH, x_HH = self.DWT(x)
        cat = torch.cat([x_LL, x_HL, x_LH], dim=1)
        out = self.Conv(cat)

        return out


class Wave_Guidance_Module(nn.Module):

    def __init__(self, encoder_channels, wave_channels, **kwargs):
        super().__init__()

        self.conv_1 = BasicConv2d(encoder_channels, wave_channels, kernel_size=1, padding=0)

        self.conv = nn.Sequential(
            BasicConv2d(wave_channels, encoder_channels // 4, kernel_size=3, padding=1),
            BasicConv2d(encoder_channels // 4, encoder_channels, kernel_size=1, padding=0), )

        self.conv_encoder_and_wave = BasicConv2d(encoder_channels + wave_channels, encoder_channels, kernel_size=1,
                                                 padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_input, wave_input):
        en = self.conv_1(encoder_input)
        feat = torch.mul(en, wave_input)
        conv = self.conv(feat)
        att = self.sigmoid(conv)
        encoder_att = encoder_input + torch.mul(encoder_input, att)
        cat = torch.cat([encoder_att, wave_input], dim=1)  # encoder_channel + 8
        out = self.conv_encoder_and_wave(cat)

        return out


class Wave_Edge_Guidance_Module(nn.Module):

    def __init__(self, encoder_channels, wave_channels, edge_channels, **kwargs):
        super().__init__()

        self.conv_encoder_to_wave_channels = BasicConv2d(encoder_channels, wave_channels, kernel_size=1, padding=0)

        self.conv_encoder_to_edge_channels = BasicConv2d(encoder_channels, edge_channels, kernel_size=1, padding=0)

        self.conv_encoder_and_wave = nn.Sequential(
            BasicConv2d(wave_channels, encoder_channels // 4, kernel_size=3, padding=1),
            BasicConv2d(encoder_channels // 4, encoder_channels, kernel_size=1, padding=0), )

        self.conv_encoder_and_edge = nn.Sequential(
            BasicConv2d(edge_channels, edge_channels // 2, kernel_size=3, padding=1),
            BasicConv2d(edge_channels // 2, edge_channels, kernel_size=1, padding=0), )

        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_input, wave_input, edge_input):
        encoder_to_wave = self.conv_encoder_to_wave_channels(encoder_input)
        wave = torch.mul(encoder_to_wave, wave_input)
        wave_att = self.conv_encoder_and_wave(wave)
        wave_att = self.sigmoid(wave_att)
        encoder_wave_att = encoder_input + torch.mul(encoder_input, wave_att)

        encoder_to_edge = self.conv_encoder_to_edge_channels(encoder_input)
        edge = torch.mul(encoder_to_edge, edge_input)
        edge_att = self.conv_encoder_and_edge(edge)
        edge_att = self.sigmoid(edge_att)
        encoder_edge_att = torch.mul(encoder_to_edge, edge_att)

        out = torch.cat([encoder_wave_att, encoder_edge_att], dim=1)  

        return out

class Encoder_PFM(nn.Module):

    def __init__(self, channel_1, channel_2,channel_3,channel_4,channel_5,**kwargs):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv_1 = BasicConv2d(channel_1 + channel_2, channel_2, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(channel_2 + channel_3, channel_3, kernel_size=3, padding=1)
        self.conv_3 = BasicConv2d(channel_3 + channel_4, channel_4, kernel_size=3, padding=1)
        self.conv_4 = BasicConv2d(channel_4 + channel_5, channel_5, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, input_1, input_2, input_3, input_4, input_5):

        input_1 = self.pool(input_1)
        input_1_2 = torch.cat([input_1,input_2], dim =1)
        input_1_2 = self.conv_1(input_1_2)

        input_1_2 = self.pool(input_1_2)
        input_1_2_3 = torch.cat([input_1_2, input_3], dim=1)
        input_1_2_3 = self.conv_2(input_1_2_3)

        input_1_2_3 = self.pool(input_1_2_3)
        input_1_2_3_4 = torch.cat([input_1_2_3, input_4], dim=1)
        input_1_2_3_4 = self.conv_3(input_1_2_3_4)

        input_1_2_3_4 = self.pool(input_1_2_3_4)
        input_1_2_3_4_5 = torch.cat([input_1_2_3_4, input_5], dim=1)
        input_1_2_3_4_5 = self.conv_4(input_1_2_3_4_5)

        out = input_1_2_3_4_5

        return out


class Decoder_PFM(nn.Module):

    def __init__(self, channel_1, channel_2,channel_3,channel_4,num_classes=4, **kwargs):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv_1 = BasicConv2d(channel_1 + channel_2, channel_2, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(channel_2 + channel_3, channel_3, kernel_size=3, padding=1)
        self.conv_3 = BasicConv2d(channel_3 + channel_4, channel_4, kernel_size=3, padding=1)

        self.conv11 = nn.Sequential(
            nn.Conv2d(32, num_classes, 1, padding=0),
            nn.Sigmoid(), )
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_1, input_2, input_3, input_4):

        input_1 = F.interpolate(input_1, scale_factor = 2, mode='bilinear')
        input_1_2 = torch.cat([input_1,input_2], dim =1)
        input_1_2 = self.conv_1(input_1_2)

        input_1_2 = F.interpolate(input_1_2, scale_factor =2, mode='bilinear')
        input_1_2_3 = torch.cat([input_1_2, input_3], dim=1)
        input_1_2_3 = self.conv_2(input_1_2_3)

        input_1_2_3 = F.interpolate(input_1_2_3, scale_factor =2, mode='bilinear')
        input_1_2_3_4 = torch.cat([input_1_2_3, input_4], dim=1)
        input_1_2_3_4 = self.conv_3(input_1_2_3_4)

        out = self.conv11(input_1_2_3_4)
        return out

class EDM(nn.Module):

    def __init__(self, input_channels,out_channels, **kwargs):
        super().__init__()
        self.edge = nn.Sequential(
            BasicConv2d(input_channels, input_channels*2, kernel_size=3, padding=1),
            BasicConv2d(input_channels*2, input_channels*2, kernel_size = 3, padding = 1),
            BasicConv2d(input_channels*2, out_channels, kernel_size=3, padding=1),)

        self.edge_conv = nn.Conv2d(out_channels, 1, 1, padding=0)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, input):
        edge = self.edge(input)
        edge_map = self.edge_conv(edge)
        edge_map = self.Sigmoid(edge_map)
        return edge,edge_map
    
    
    
    
class MWGNet(nn.Module): 
    def __init__(self,in_channels=3, num_classes=4):
        super(MWGNet, self).__init__()

        self.vgg = VGG()

        self.EDM = EDM(64, 64)

        self.pool_2x = nn.MaxPool2d(2)
        
        self.conv_0 = nn.Sequential(nn.Conv2d(in_channels, 3, 1, padding=0))

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(256 + 512 + 64, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(128 + 256 + 64, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(64 + 128 + 64, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(32 + 64, 32)

        self.conv10 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), )

        self.conv11 = nn.Sequential(
            nn.Conv2d(32, 1, 1, padding=0),
            nn.Sigmoid(), )

        self.DWT_block_1 = DWT_block_1(3*in_channels, 16)
        self.DWT_block_2 = DWT_block_2(3*in_channels, 32)
        self.DWT_block_3 = DWT_block_3(3*in_channels, 64)
        self.DWT_block_4 = DWT_block_4(3*in_channels, 128)

        self.WGM_1 = Wave_Guidance_Module(64, 16)  
        self.WGM_2 = Wave_Guidance_Module(128, 32)  
        self.WGM_3 = Wave_Guidance_Module(256, 64)  
        self.WGM_4 = Wave_Guidance_Module(512, 128)  

        self.WEGM_1 = Wave_Edge_Guidance_Module(64, 16, 64)  
        self.WEGM_2 = Wave_Edge_Guidance_Module(128, 32, 64)  
        self.WEGM_3 = Wave_Edge_Guidance_Module(256, 64, 64) 

        self.En_PFM = Encoder_PFM(64, 128, 256, 512, 512)

        self.De_PFM = Decoder_PFM(256, 128, 64, 32, num_classes)


    def forward(self, x):
        wave_1 = self.DWT_block_1(x)
        wave_2 = self.DWT_block_2(x)
        wave_3 = self.DWT_block_3(x)
        wave_4 = self.DWT_block_4(x)
        
        c0 = self.conv_0(x) 

        c1 = self.vgg.conv1(c0)
        p1 = self.pool_2x(c1)
        p1 = self.WGM_1(p1, wave_1)

        Edge, Edge_map = self.EDM(c1)
        Edge_256 = self.pool_2x(Edge)
        Edge_128 = self.pool_2x(Edge_256)
        Edge_64 = self.pool_2x(Edge_128)

        c2 = self.vgg.conv2(p1)
        p2 = self.pool_2x(c2)
        p2 = self.WGM_2(p2, wave_2)

        c3 = self.vgg.conv3(p2)
        p3 = self.pool_2x(c3)
        p3 = self.WGM_3(p3, wave_3)

        c4 = self.vgg.conv4_1(p3)
        p4 = self.pool_2x(c4)
        p4 = self.WGM_4(p4, wave_4)

        c5 = self.vgg.conv5_1(p4)

        Encoder = self.En_PFM(c1, c2, c3, c4, c5)

        up_6 = self.up6(Encoder) 
        c6 = self.WEGM_3(up_6, wave_3, Edge_64)
        merge6 = torch.cat([c6, c4], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        c7 = self.WEGM_2(up_7, wave_2, Edge_128)
        merge7 = torch.cat([c7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        c8 = self.WEGM_1(up_8, wave_1, Edge_256)
        merge8 = torch.cat([c8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        out = self.De_PFM(c6, c7, c8, c9)


        return out
        # return Edge_map, out
    


if __name__=='__main__': 
    import os
    import sys
    CUDA_LAUNCH_BLOCKING=1
    os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    items = current_dir.split('/')
    for item in items:
        if 'v2024' in item or  'V2024' in item:
            obj_name = item
    obj_name = os.path.abspath('./DWT_2023/{}'.format(obj_name)) # add current object name to sys
    sys.path.append(obj_name)

    # from models.sota_models.MWG_Net.VGGNET import VGG

    model = MWGNet(in_channels=3, num_classes=4).cuda()
    
    input = torch.randn(1,3,512,512)
    
    input = input.cuda()
    
    out = model(input)
    
    print(out)