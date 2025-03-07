#From: https://github.com/LeeJunHyun/Image_Segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *

class U_Net(nn.Module):
    def __init__(self, img_ch :  int = 3,output_ch : int= 1, channels=[64, 128, 256, 512, 1024], act_func=None, drop_rate=0.0):
        super(U_Net,self).__init__()
        
        if act_func is None:
            act_func = nn.ReLU
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=channels[0], act=act_func, drop_rate=drop_rate)
        self.Conv2 = conv_block(ch_in=channels[0],ch_out=channels[1], act=act_func, drop_rate=drop_rate)
        self.Conv3 = conv_block(ch_in=channels[1],ch_out=channels[2], act=act_func, drop_rate=drop_rate)
        self.Conv4 = conv_block(ch_in=channels[2],ch_out=channels[3], act=act_func, drop_rate=drop_rate)
        self.Conv5 = conv_block(ch_in=channels[3],ch_out=channels[4], act=act_func, drop_rate=drop_rate)

        self.Up5 = up_conv(ch_in=channels[4],ch_out=channels[3], act=act_func, drop_rate=drop_rate)
        self.Up_conv5 = conv_block(ch_in=channels[4], ch_out=channels[3], act=act_func, drop_rate=drop_rate)

        self.Up4 = up_conv(ch_in=channels[3],ch_out=channels[2], act=act_func, drop_rate=drop_rate)
        self.Up_conv4 = conv_block(ch_in=channels[3], ch_out=channels[2], act=act_func, drop_rate=drop_rate)
        
        self.Up3 = up_conv(ch_in=channels[2],ch_out=channels[1], act=act_func, drop_rate=drop_rate)
        self.Up_conv3 = conv_block(ch_in=channels[2], ch_out=channels[1], act=act_func, drop_rate=drop_rate)
        
        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[0], act=act_func, drop_rate=drop_rate)
        self.Up_conv2 = conv_block(ch_in=channels[1], ch_out=channels[0], act=act_func, drop_rate=drop_rate)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1, channels=[64, 128, 256, 512, 1024], act=None, drop_rate=0.0):
        super(AttU_Net,self).__init__()
        
        if act is None:
            act = nn.ReLU
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=channels[0], act=act, drop_rate=drop_rate)
        self.Conv2 = conv_block(ch_in=channels[0],ch_out=channels[1], act=act, drop_rate=drop_rate)
        self.Conv3 = conv_block(ch_in=channels[1],ch_out=channels[2], act=act, drop_rate=drop_rate)
        self.Conv4 = conv_block(ch_in=channels[2],ch_out=channels[3], act=act, drop_rate=drop_rate)
        self.Conv5 = conv_block(ch_in=channels[3],ch_out=channels[4], act=act, drop_rate=drop_rate)

        self.Up5 = up_conv(ch_in=channels[4],ch_out=channels[3], act=act, drop_rate=drop_rate)
        self.Att5 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=channels[3] // 2)
        self.Up_conv5 = conv_block(ch_in=channels[4], ch_out=channels[3], act=act, drop_rate=drop_rate)

        self.Up4 = up_conv(ch_in=channels[3],ch_out=channels[2], act=act, drop_rate=drop_rate)
        self.Att4 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[2] // 2)
        self.Up_conv4 = conv_block(ch_in=channels[3], ch_out=channels[2], act=act, drop_rate=drop_rate)
        
        self.Up3 = up_conv(ch_in=channels[2],ch_out=channels[1], act=act, drop_rate=drop_rate)
        self.Att3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[1] // 2)
        self.Up_conv3 = conv_block(ch_in=channels[2], ch_out=channels[1], act=act, drop_rate=drop_rate)
        
        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[0], act=act, drop_rate=drop_rate)
        self.Att2 = Attention_block(F_g=channels[0],F_l=channels[0],F_int=channels[0] // 2)
        self.Up_conv2 = conv_block(ch_in=channels[1], ch_out=channels[0], act=act, drop_rate=drop_rate)

        self.Conv_1x1 = nn.Conv2d(channels[0],output_ch,kernel_size=1,stride=1,padding=0)



    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net4(nn.Module):
    def __init__(self,img_ch=1,output_ch=1, channels=[64, 128, 256, 512], act=None, drop_rate=0.0):
        super(AttU_Net4,self).__init__()
        
        if act is None:
            act = nn.ReLU
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=channels[0], act=act, drop_rate=drop_rate)
        self.Conv2 = conv_block(ch_in=channels[0],ch_out=channels[1], act=act, drop_rate=drop_rate)
        self.Conv3 = conv_block(ch_in=channels[1],ch_out=channels[2], act=act, drop_rate=drop_rate)
        self.Conv4 = conv_block(ch_in=channels[2],ch_out=channels[3], act=act, drop_rate=drop_rate)

        self.Up4 = up_conv(ch_in=channels[3],ch_out=channels[2], act=act, drop_rate=drop_rate)
        self.Att4 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[2] // 2)
        self.Up_conv4 = conv_block(ch_in=channels[3], ch_out=channels[2], act=act, drop_rate=drop_rate)
        
        self.Up3 = up_conv(ch_in=channels[2],ch_out=channels[1], act=act, drop_rate=drop_rate)
        self.Att3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[1] // 2)
        self.Up_conv3 = conv_block(ch_in=channels[2], ch_out=channels[1], act=act, drop_rate=drop_rate)
        
        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[0], act=act, drop_rate=drop_rate)
        self.Att2 = Attention_block(F_g=channels[0],F_l=channels[0],F_int=channels[0] // 2)
        self.Up_conv2 = conv_block(ch_in=channels[1], ch_out=channels[0], act=act, drop_rate=drop_rate)

        self.Conv_1x1 = nn.Conv2d(channels[0],output_ch,kernel_size=1,stride=1,padding=0)



    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        
        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
    