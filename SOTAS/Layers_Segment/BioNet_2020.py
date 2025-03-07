import torch
import torch.nn as nn
import torchvision.models as models

"""@ARTICLE{Zhang2020AutoSeg,

  author={Zhang, Huihong and Yang, Jianlong and Zhou, Kang and Li, Fei and Hu, Yan and Zhao, Yitian and Zheng, Ce and Zhang, Xiulan and Liu, Jiang},

  journal={IEEE Journal of Biomedical and Health Informatics}, 

  title={Automatic Segmentation and Visualization of Choroid in OCT with Knowledge Infused Deep Learning}, 

  year={2020},

  volume={24},

  number={12},

  pages={3408-3420},

  doi={10.1109/JBHI.2020.3023144}}
  """

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.maxpool(e1))
        e3 = self.enc3(self.maxpool(e2))
        e4 = self.enc4(self.maxpool(e3))
        
        # Decoder
        d4 = self.up4(e4)
        d4 = torch.cat([e3, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([e2, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e1, d2], dim=1)
        d2 = self.dec2(d2)
        
        return self.final(d2)

class BioRegularization(nn.Module):
    def __init__(self, in_channels):
        super(BioRegularization, self).__init__()
        # Initialize first conv layer to handle arbitrary input channels
        self.init_conv = nn.Conv2d(in_channels, 3, kernel_size=1)
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)
        
    def forward(self, x):
        # Convert input channels to 3 channels for ResNet
        x = self.init_conv(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BioNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, gms_channels=2):
        super(BioNet, self).__init__()
        
        # Global Multilayers Segmentation Module (GMS)
        self.gms = UNet(in_channels, out_channels=gms_channels)
        
        # Local Choroid Segmentation Module (LCS)
        self.lcs = UNet(in_channels + gms_channels, out_channels=num_classes)
        
        # Biomarker Regularization Module
        self.bio = BioRegularization(in_channels=in_channels+num_classes)
        
        # 保存配置参数
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.gms_channels = gms_channels
        
    def forward(self, x):
        # Global Multilayers Segmentation
        gms_out = self.gms(x)
        
        # Concatenate input with GMS output
        lcs_input = torch.cat([x, gms_out], dim=1)
        
        # Local Choroid Segmentation
        seg_pred = self.lcs(lcs_input)
        
        # Biomarker Regularization
        bio_input = torch.cat([x, seg_pred], dim=1)
        bio_out = self.bio(bio_input)
        
        return seg_pred, gms_out, bio_out

if __name__ == "__main__":
    # 测试参数  
    in_channels = 3 
    num_classes = 2
    gms_channels = 7
    
    # 创建测试输入
    x = torch.randn(2, in_channels, 256, 256)
    
    # 初始化模型
    model = BioNet(
        in_channels=in_channels,
        num_classes=num_classes,
        gms_channels=gms_channels
    )
    
    # 测试前向传播
    seg_pred, gms_out, bio_out = model(x)
    
    # 打印输出形状
    print(f"Input shape: {x.shape}")
    print(f"Segmentation output shape: {seg_pred.shape}")
    print(f"GMS output shape: {gms_out.shape}")
    print(f"Biomarker output shape: {bio_out.shape}")
    
  
