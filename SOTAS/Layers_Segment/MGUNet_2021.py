"""@article{li2021multi,
  title={Multi-scale GCN-assisted two-stage network for joint segmentation of retinal layers and discs in peripapillary OCT images},
  author={Li, Jiaxuan and Jin, Peiyao and Zhu, Jianfeng and Zou, Haidong and Xu, Xun and Tang, Min and Zhou, Minwen and Gan, Yu and He, Jiangnan and Ling, Yuye and others},
  journal={Biomedical Optics Express},
  volume={12},
  number={4},
  pages={2204--2220},
  year={2021},
  publisher={Optica Publishing Group}
}"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import init

# 添加项目根目录到路径
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(project_root)

# # 从utils导入所需组件
# from models.utils.utils import Basconv, UnetConv, UnetUp, UnetUp4, GloRe_Unit
# from models.utils.init_weights import init_weights

# 基础组件定义
class Basconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Basconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UnetConv(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm=True):
        super(UnetConv, self).__init__()
        
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, is_deconv=True):
        super(UnetUp, self).__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 1)
            )
            
        self.conv = UnetConv(in_channels, out_channels, True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UnetUp4(nn.Module):
    def __init__(self, in_channels, out_channels, is_deconv=True):
        super(UnetUp4, self).__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=4)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=4),
                nn.Conv2d(in_channels, out_channels, 1)
            )
            
        self.conv = UnetConv(in_channels, out_channels, True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class GloRe_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1):
        super(GloRe_Unit, self).__init__()
        self.N = in_channels
        self.M = out_channels
        
        # 降维投影
        self.conv_state = nn.Conv2d(self.N, self.M, kernel_size=1)
        self.conv_proj = nn.Conv2d(self.N, self.M, kernel_size=1)
        # 升维投影
        self.conv_extend = nn.Conv2d(self.M, self.N, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()
        
        # 降维投影
        x_state = self.conv_state(x)  # [n, M, h, w]
        x_proj = self.conv_proj(x)    # [n, M, h, w]
        
        # 重塑维度
        hw = h * w
        x_state = x_state.view(n, self.M, -1)  # [n, M, hw]
        x_proj = x_proj.view(n, self.M, -1)    # [n, M, hw]
        
        # 计算注意力
        x_rproj = torch.bmm(x_state, x_proj.transpose(1, 2))  # [n, M, M]
        x_rproj = x_rproj / (hw ** 0.5)  # 缩放因子
        x_rproj = F.softmax(x_rproj, dim=2)
        
        # 特征聚合
        x_rstate = torch.bmm(x_rproj, x_proj)  # [n, M, hw]
        
        # 恢复空间维度
        x_rstate = x_rstate.view(n, self.M, h, w)
        
        # 升维投影
        out = x + self.conv_extend(x_rstate)
        
        return out

class  MGR_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MGR_Module, self).__init__()

        self.conv0_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou0 = nn.Sequential(OrderedDict([("GCN%02d" % i, GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv1_1 = Basconv(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.conv1_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou1 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv2_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.conv2_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou2 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))

        self.conv3_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.conv3_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou3 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))
        
        self.f1 = Basconv(in_channels=4*out_channels, out_channels=in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)

        self.x0 = self.conv0_1(x)
        self.g0 = self.glou0(self.x0)

        self.x1 = self.conv1_2(self.pool1(self.conv1_1(x)))
        self.g1 = self.glou1(self.x1)
        self.layer1 = F.interpolate(self.g1, size=(h, w), mode='bilinear', align_corners=True)

        self.x2 = self.conv2_2(self.pool2(self.conv2_1(x)))
        self.g2 = self.glou2(self.x2)
        self.layer2 = F.interpolate(self.g2, size=(h, w), mode='bilinear', align_corners=True)

        self.x3 = self.conv3_2(self.pool3(self.conv3_1(x)))
        self.g3= self.glou3(self.x3)
        self.layer3 = F.interpolate(self.g3, size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat([self.g0, self.layer1, self.layer2, self.layer3], 1)

        return self.f1(out)



class MGUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=11, feature_scale=4, is_deconv=True, is_batchnorm=True):  ##########
        super(MGUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # encoder
        self.conv1 = UnetConv(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=4)

        self.conv3 = UnetConv(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=4)

        self.mgb =  MGR_Module(filters[2], filters[3])

        self.center = UnetConv(filters[2], filters[3], self.is_batchnorm)

        # decoder
        self.up_concat3 = UnetUp4(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp4(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp(filters[1], filters[0], self.is_deconv)

        # final conv
        self.final_1 = nn.Conv2d(filters[0], num_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  
        maxpool1 = self.maxpool1(conv1)  
        conv2 = self.conv2(maxpool1)  
        maxpool2 = self.maxpool2(conv2)  
        conv3 = self.conv3(maxpool2) 
        maxpool3 = self.maxpool3(conv3)  
        feat_sum = self.mgb(maxpool3)
        center = self.center(feat_sum)  
        up3 = self.up_concat3(center, conv3) 
        up2 = self.up_concat2(up3, conv2) 
        up1 = self.up_concat1(up2, conv1)
        final_1 = self.final_1(up1)

        return final_1


class MGUNet_2(nn.Module):
    def __init__(self, in_channels=1, num_classes=11, feature_scale=4, is_deconv=True, is_batchnorm=True):  ##########
        super(MGUNet_2, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # encoder
        self.conv1 = UnetConv(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.mgb =  MGR_Module(filters[2], filters[3])

        self.center = UnetConv(filters[2], filters[3], self.is_batchnorm)

        # decoder
        self.up_concat3 = UnetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp(filters[1], filters[0], self.is_deconv)

        # final conv
        self.final_1 = nn.Conv2d(filters[0], num_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  
        maxpool1 = self.maxpool1(conv1) 
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)  
        conv3 = self.conv3(maxpool2)  
        maxpool3 = self.maxpool3(conv3)  
        feat_sum = self.mgb(maxpool3) 
        center = self.center(feat_sum)  
        up3 = self.up_concat3(center, conv3) 
        up2 = self.up_concat2(up3, conv2) 
        up1 = self.up_concat1(up2, conv1)
        final_1 = self.final_1(up1)

        return final_1

import torch.nn as nn
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    
    
    

if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型和测试数据
    model = MGUNet_1(in_channels=1, num_classes=9).cuda()
    x = torch.randn(4, 1, 352, 352).cuda()
    
    try:
        # 前向传播测试
        with torch.no_grad():
            out = model(x)
            print(f"\nInput shape: {x.shape}")
            print(f"Output shape: {out.shape}")
    except Exception as e:
        print(f"\nError during forward pass: {str(e)}")
        import traceback
        traceback.print_exc()