import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
items = current_dir.split('/')
for item in items:
    if 'v2024' in item or 'V2024' in item:
        obj_name = item
# 修改这行，使用项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '../../'))  # 回退到v20240614目录
sys.path.append(project_root)  # 添加项目根目录到Python路径


from basics.res2net import res2net50_v1b_26w_4s
import torchvision
"""
Official repository of "Automatic Polyp Segmentation via Multi-scale Subtraction Network" MICCAI-2021.
Code in https://github.com/Xiaoqi-Zhao-DLUT/MSNet-M2SNet;

New paper: M2SNet: Multi-scale in Multi-scale Subtraction Network for Medical Image Segmentation
"""

"""
@inproceedings{zhao2021automatic,
  title={Automatic polyp segmentation via multi-scale subtraction network},
  author={Zhao, Xiaoqi and Zhang, Lihe and Lu, Huchuan},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2021: 24th International Conference, Strasbourg, France, September 27--October 1, 2021, Proceedings, Part I 24},
  pages={120--130},
  year={2021},
  organization={Springer}
}
"""

class CNN1(nn.Module):
    def __init__(self, channel, map_size, pad):
        super(CNN1, self).__init__()
        # 修改权重初始化方式
        self.conv = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=map_size,
            padding=pad,
            groups=channel  # 使用分组卷积
        )
        self.norm = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class MSNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, in_channels=3, num_classes=1):
        super(MSNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=False)
        
        # 修改第一层卷积以适应不同的输入通道数
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        self.x5_dem_1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_x4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x5_x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.x5_dem_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))

        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_dem_5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64, num_classes, kernel_size=3, padding=1))

    def forward(self, x):
        input = x

        # '''
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x1 = self.resnet.maxpool(x)      # bs, 64, 88, 88
        # ---- low-level features ----
        x2 = self.resnet.layer1(x1)      # bs, 256, 88, 88
        x3 = self.resnet.layer2(x2)     # bs, 512, 44, 44
        x4 = self.resnet.layer3(x3)     # bs, 1024, 22, 22
        x5 = self.resnet.layer4(x4)     # bs, 2048, 11, 11
        # '''


        x5_dem_1 = self.x5_dem_1(x5)
        x4_dem_1 = self.x4_dem_1(x4)
        x3_dem_1 = self.x3_dem_1(x3)
        x2_dem_1 = self.x2_dem_1(x2)

        x5_4 = self.x5_x4(abs(F.upsample(x5_dem_1,size=x4.size()[2:], mode='bilinear')-x4_dem_1))
        x4_3 = self.x4_x3(abs(F.upsample(x4_dem_1,size=x3.size()[2:], mode='bilinear')-x3_dem_1))
        x3_2 = self.x3_x2(abs(F.upsample(x3_dem_1,size=x2.size()[2:], mode='bilinear')-x2_dem_1))
        x2_1 = self.x2_x1(abs(F.upsample(x2_dem_1,size=x1.size()[2:], mode='bilinear')-x1))


        x5_4_3 = self.x5_x4_x3(abs(F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear') - x4_3))
        x4_3_2 = self.x4_x3_x2(abs(F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear') - x3_2))
        x3_2_1 = self.x3_x2_x1(abs(F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear') - x2_1))


        x5_4_3_2 = self.x5_x4_x3_x2(abs(F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear') - x4_3_2))
        x4_3_2_1 = self.x4_x3_x2_x1(abs(F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear') - x3_2_1))

        x5_dem_4 = self.x5_dem_4(x5_4_3_2)
        x5_4_3_2_1 = self.x5_x4_x3_x2_x1(abs(F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear') - x4_3_2_1))

        level4 = x5_4
        level3 = self.level3(x4_3 + x5_4_3)
        level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2)
        level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)

        x5_dem_5 = self.x5_dem_5(x5)
        output4 = self.output4(F.upsample(x5_dem_5,size=level4.size()[2:], mode='bilinear') + level4)
        output3 = self.output3(F.upsample(output4,size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3,size=level2.size()[2:], mode='bilinear') + level2)
        output1 = self.output1(F.upsample(output2,size=level1.size()[2:], mode='bilinear') + level1)

        output = F.upsample(output1, size=input.size()[2:], mode='bilinear')
        if self.training:
            return output
        return output


class M2SNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, in_channels=3, num_classes=1):
        super(M2SNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=False)
        
        # 修改第一层卷积以适应不同的输入通道数
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # 修改CNN1的初始化
        self.conv_3 = CNN1(64, 3, 1)
        self.conv_5 = CNN1(64, 5, 2)

        # 打印调试信息
        # print(f"\nInitializing M2SNet:")
        # print(f"Input channels: {in_channels}")
        # print(f"Output classes: {num_classes}")
        
        self.x5_dem_1 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
        )
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_x4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x5_x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.x5_dem_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))

        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_dem_5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64, num_classes, kernel_size=3, padding=1))

    def forward(self, x):
        # 添加shape检查
        # print(f"\nInput shape: {x.shape}")
        
        input = x
        x = self.resnet.conv1(x)
        # print(f"After conv1 shape: {x.shape}")
        
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x1 = self.resnet.maxpool(x)
        # print(f"After maxpool shape: {x1.shape}")
        
        # 继续前向传播...
        x2 = self.resnet.layer1(x1)
        # print(f"After layer1 shape: {x2.shape}")
        
        x3 = self.resnet.layer2(x2)
        # print(f"After layer2 shape: {x3.shape}")
        
        x4 = self.resnet.layer3(x3)
        # print(f"After layer3 shape: {x4.shape}")
        
        x5 = self.resnet.layer4(x4)
        # print(f"After layer4 shape: {x5.shape}")
        
        x5_dem_1 = self.x5_dem_1(x5)
        x4_dem_1 = self.x4_dem_1(x4)
        x3_dem_1 = self.x3_dem_1(x3)
        x2_dem_1 = self.x2_dem_1(x2)

        x5_dem_1_up = F.upsample(x5_dem_1, size=x4.size()[2:], mode='bilinear')
        x5_dem_1_up_map1 = self.conv_3(x5_dem_1_up)
        x4_dem_1_map1 = self.conv_3(x4_dem_1)
        x5_dem_1_up_map2 = self.conv_5(x5_dem_1_up)
        x4_dem_1_map2 = self.conv_5(x4_dem_1)
        x5_4 = self.x5_x4(
            abs(x5_dem_1_up - x4_dem_1)+abs(x5_dem_1_up_map1-x4_dem_1_map1)+abs(x5_dem_1_up_map2-x4_dem_1_map2))


        x4_dem_1_up = F.upsample(x4_dem_1, size=x3.size()[2:], mode='bilinear')
        x4_dem_1_up_map1 = self.conv_3(x4_dem_1_up)
        x3_dem_1_map1 = self.conv_3(x3_dem_1)
        x4_dem_1_up_map2 = self.conv_5(x4_dem_1_up)
        x3_dem_1_map2 = self.conv_5(x3_dem_1)
        x4_3 = self.x4_x3(
            abs(x4_dem_1_up - x3_dem_1)+abs(x4_dem_1_up_map1-x3_dem_1_map1)+abs(x4_dem_1_up_map2-x3_dem_1_map2) )


        x3_dem_1_up = F.upsample(x3_dem_1, size=x2.size()[2:], mode='bilinear')
        x3_dem_1_up_map1 = self.conv_3(x3_dem_1_up)
        x2_dem_1_map1 = self.conv_3(x2_dem_1)
        x3_dem_1_up_map2 = self.conv_5(x3_dem_1_up)
        x2_dem_1_map2 = self.conv_5(x2_dem_1)
        x3_2 = self.x3_x2(
            abs(x3_dem_1_up - x2_dem_1)+abs(x3_dem_1_up_map1-x2_dem_1_map1)+abs(x3_dem_1_up_map2-x2_dem_1_map2) )


        x2_dem_1_up = F.upsample(x2_dem_1, size=x1.size()[2:], mode='bilinear')
        x2_dem_1_up_map1 = self.conv_3(x2_dem_1_up)
        x1_map1 = self.conv_3(x1)
        x2_dem_1_up_map2 = self.conv_5(x2_dem_1_up)
        x1_map2 = self.conv_5(x1)
        x2_1 = self.x2_x1(abs(x2_dem_1_up - x1)+abs(x2_dem_1_up_map1-x1_map1)+abs(x2_dem_1_up_map2-x1_map2) )


        x5_4_up = F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear')
        x5_4_up_map1 = self.conv_3(x5_4_up)
        x4_3_map1 = self.conv_3(x4_3)
        x5_4_up_map2 = self.conv_5(x5_4_up)
        x4_3_map2 = self.conv_5(x4_3)
        x5_4_3 = self.x5_x4_x3(abs(x5_4_up - x4_3) +abs(x5_4_up_map1-x4_3_map1)+abs(x5_4_up_map2-x4_3_map2))


        x4_3_up = F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear')
        x4_3_up_map1 = self.conv_3(x4_3_up)
        x3_2_map1 = self.conv_3(x3_2)
        x4_3_up_map2 = self.conv_5(x4_3_up)
        x3_2_map2 = self.conv_5(x3_2)
        x4_3_2 = self.x4_x3_x2(abs(x4_3_up - x3_2)+abs(x4_3_up_map1-x3_2_map1)+abs(x4_3_up_map2-x3_2_map2) )


        x3_2_up = F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear')
        x3_2_up_map1 = self.conv_3(x3_2_up)
        x2_1_map1 = self.conv_3(x2_1)
        x3_2_up_map2 = self.conv_5(x3_2_up)
        x2_1_map2 = self.conv_5(x2_1)
        x3_2_1 = self.x3_x2_x1(abs(x3_2_up - x2_1)+abs(x3_2_up_map1-x2_1_map1)+abs(x3_2_up_map2-x2_1_map2) )


        x5_4_3_up = F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear')
        x5_4_3_up_map1 = self.conv_3(x5_4_3_up)
        x4_3_2_map1 = self.conv_3(x4_3_2)
        x5_4_3_up_map2 = self.conv_5(x5_4_3_up)
        x4_3_2_map2 = self.conv_5(x4_3_2)
        x5_4_3_2 = self.x5_x4_x3_x2(
            abs(x5_4_3_up - x4_3_2)+abs(x5_4_3_up_map1-x4_3_2_map1)+abs(x5_4_3_up_map2-x4_3_2_map2) )


        x4_3_2_up = F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear')
        x4_3_2_up_map1 = self.conv_3(x4_3_2_up)
        x3_2_1_map1 = self.conv_3(x3_2_1)
        x4_3_2_up_map2 = self.conv_5(x4_3_2_up)
        x3_2_1_map2 = self.conv_5(x3_2_1)
        x4_3_2_1 = self.x4_x3_x2_x1(
            abs(x4_3_2_up - x3_2_1) +abs(x4_3_2_up_map1-x3_2_1_map1)+abs(x4_3_2_up_map2-x3_2_1_map2))


        x5_dem_4 = self.x5_dem_4(x5_4_3_2)
        x5_dem_4_up = F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear')
        x5_dem_4_up_map1 = self.conv_3(x5_dem_4_up)
        x4_3_2_1_map1 = self.conv_3(x4_3_2_1)
        x5_dem_4_up_map2 = self.conv_5(x5_dem_4_up)
        x4_3_2_1_map2 = self.conv_5(x4_3_2_1)
        x5_4_3_2_1 = self.x5_x4_x3_x2_x1(
            abs(x5_dem_4_up - x4_3_2_1)+abs(x5_dem_4_up_map1-x4_3_2_1_map1)+abs(x5_dem_4_up_map2-x4_3_2_1_map2) )

        level4 = x5_4
        level3 = self.level3(x4_3 + x5_4_3)
        level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2)
        level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)

        x5_dem_5 = self.x5_dem_5(x5)
        output4 = self.output4(F.upsample(x5_dem_5,size=level4.size()[2:], mode='bilinear') + level4)
        output3 = self.output3(F.upsample(output4,size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3,size=level2.size()[2:], mode='bilinear') + level2)
        output1 = self.output1(F.upsample(output2,size=level1.size()[2:], mode='bilinear') + level1)
        output = F.upsample(output1, size=input.size()[2:], mode='bilinear')
        if self.training:
            return output
        return output
    
    
class LossNet(torch.nn.Module):
    def __init__(self, resize=True):
        super(LossNet, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target

        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.mse_loss(x, y)
        return loss




if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型和测试数据
    model = M2SNet(in_channels=1, num_classes=9).cuda()
    x = torch.randn(4, 1, 352, 352).cuda()
    
    # 打印模型结构
    print("\nModel structure:")
    print(model)
    
    # 前向传播测试
    try:
        with torch.no_grad():
            out = model(x)
            print(f"\nOutput shape: {out.shape}")
    except Exception as e:
        print(f"\nError during forward pass: {str(e)}")
        import traceback
        traceback.print_exc()