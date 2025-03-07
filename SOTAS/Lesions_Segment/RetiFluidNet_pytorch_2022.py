import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
Pytorch implementation of original tensorflow version:
https://github.com/aidialab/retifluidnet
"""
"""
@article{rasti2022retifluidnet,
  title={RetiFluidNet: a self-adaptive and multi-attention deep convolutional network for retinal OCT fluid segmentation},
  author={Rasti, Reza and Biglari, Armin and Rezapourian, Mohammad and Yang, Ziyun and Farsiu, Sina},
  journal={IEEE Transactions on Medical Imaging},
  volume={42},
  number={5},
  pages={1413--1423},
  year={2022},
  publisher={IEEE}
}
"""
class RetiFluidNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, input_shape=(256, 256, 1)):
        super(RetiFluidNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # 计算每个阶段的通道数
        base_channels = 64
        self.channel_sizes = [
            base_channels,      # stage 0
            base_channels * 2,  # stage 1
            base_channels * 4,  # stage 2
            base_channels * 8,  # stage 3
            base_channels * 16  # stage 4
        ]
        
        # 为每个阶段创建对应通道数的alpha和beta层
        self.alpha_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            for channels in self.channel_sizes
        ])
        self.beta_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            for channels in self.channel_sizes
        ])
        
        # 初始化权重为1
        for conv in self.alpha_convs + self.beta_convs:
            nn.init.constant_(conv.weight, 1.0)
            
        # 初始化第一个卷积层
        self.initial_conv = nn.Conv2d(in_channels, 64, 3, padding=1)

    def SDA(self, x, p_scale=4, SDAblock_nb=0):
        batch, c, h, w = x.size()
        input_tensor = x
        
        # 检查通道数是否匹配
        expected_channels = self.channel_sizes[SDAblock_nb]
        assert c == expected_channels, f"Expected {expected_channels} channels, but got {c}"
        
        # MaxPool
        tensor = F.max_pool2d(x, p_scale)
        _, c, hp, wp = tensor.size()
        
        # Pixel-wise attention
        ratio1 = np.sqrt(hp * wp)
        reshaped_tensor = tensor.view(batch, c, -1)
        transposed_tensor = reshaped_tensor.permute(0, 2, 1)
        
        pixel_attn = torch.bmm(transposed_tensor, reshaped_tensor) / ratio1
        pixel_attn = F.softmax(pixel_attn, dim=-1)
        pixel_out = torch.bmm(pixel_attn, transposed_tensor)
        
        pixel_out = pixel_out.view(batch, hp, wp, c).permute(0, 3, 1, 2)
        pixel_out = self.alpha_convs[SDAblock_nb](pixel_out)
        add_1 = F.interpolate(pixel_out, size=(h, w), mode='nearest')
        
        # Channel-wise attention
        ratio2 = np.sqrt(c * c)
        channel_attn = torch.bmm(reshaped_tensor, reshaped_tensor.permute(0, 2, 1)) / ratio2
        channel_attn = F.softmax(channel_attn, dim=-1)
        channel_out = torch.bmm(channel_attn, reshaped_tensor)
        
        channel_out = channel_out.view(batch, c, hp, wp)
        channel_out = self.beta_convs[SDAblock_nb](channel_out)
        add_2 = F.interpolate(channel_out, size=(h, w), mode='nearest')
        
        mean = 0.5 * (add_1 + add_2)
        out = input_tensor + mean
        
        return out

    def encoder_block(self, x, filters, kernel_size=3, SDAblock_nb=0):
       
        
        # 先进行卷积操作
        conv = nn.Sequential(
            nn.Conv2d(x.size(1), filters, kernel_size, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        x = conv(x)
        # print(f"After conv shape: {x.shape}")
        
        # 然后应用SDA
        sda = self.SDA(x, p_scale=4, SDAblock_nb=SDAblock_nb)
        # print(f"After SDA shape: {sda.shape}")
        
        return x + sda

    def decoder_block(self, x, skip, filters, kernel_size=3, SDAblock_nb=0):
        x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = nn.Sequential(
            nn.Conv2d(x.size(1), filters, kernel_size, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size, padding=1), 
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )(x)
        sda = self.SDA(x, p_scale=4, SDAblock_nb=SDAblock_nb)
        return x + sda

    def convert_to_8_channels(self, x):
        x = F.softmax(x, dim=1)
        x = torch.argmax(x, dim=1)
        x = F.one_hot(x, num_classes=8).permute(0, 3, 1, 2).float()
        return x

    def forward(self, x):
        # Initial conv
        x = self.initial_conv(x)
        input_size = x.shape[2:]
        
        # Encoder path
        nb_filters = 64
        encoder0 = self.encoder_block(x, nb_filters, SDAblock_nb=0)
        pool0 = F.max_pool2d(encoder0, 2)
        
        encoder1 = self.encoder_block(pool0, nb_filters*2, SDAblock_nb=1) 
        pool1 = F.max_pool2d(encoder1, 2)
        
        encoder2 = self.encoder_block(pool1, nb_filters*4, SDAblock_nb=2)
        pool2 = F.max_pool2d(encoder2, 2)
        
        encoder3 = self.encoder_block(pool2, nb_filters*8, SDAblock_nb=3)
        pool3 = F.max_pool2d(encoder3, 2)
        
        encoder4 = self.encoder_block(pool3, nb_filters*16, SDAblock_nb=4)
        
        # Decoder path with size matching
        decoder4 = encoder4
        output4 = F.interpolate(decoder4, size=input_size, mode='bilinear', align_corners=True)
        output4 = nn.Conv2d(nb_filters*16, self.num_classes, 1)(output4)
        output4 = F.softmax(output4, dim=1)
        bicon_output4 = self.convert_to_8_channels(output4)
        
        decoder3 = self.decoder_block(decoder4, encoder3, nb_filters*8, SDAblock_nb=3)
        output3 = F.interpolate(decoder3, size=input_size, mode='bilinear', align_corners=True)
        output3 = nn.Conv2d(nb_filters*8, self.num_classes, 1)(output3)
        output3 = F.softmax(output3, dim=1)
        bicon_output3 = self.convert_to_8_channels(output3)
        
        decoder2 = self.decoder_block(decoder3, encoder2, nb_filters*4, SDAblock_nb=2)
        output2 = F.interpolate(decoder2, size=input_size, mode='bilinear', align_corners=True)
        output2 = nn.Conv2d(nb_filters*4, self.num_classes, 1)(output2)
        output2 = F.softmax(output2, dim=1)
        bicon_output2 = self.convert_to_8_channels(output2)
        
        decoder1 = self.decoder_block(decoder2, encoder1, nb_filters*2, SDAblock_nb=1)
        output1 = F.interpolate(decoder1, size=input_size, mode='bilinear', align_corners=True)
        output1 = nn.Conv2d(nb_filters*2, self.num_classes, 1)(output1)
        output1 = F.softmax(output1, dim=1)
        bicon_output1 = self.convert_to_8_channels(output1)
        
        decoder0 = self.decoder_block(decoder1, encoder0, nb_filters, SDAblock_nb=0)
        outputs = nn.Conv2d(nb_filters, self.num_classes, 1)(decoder0)
        main_output = F.softmax(outputs, dim=1)
        bicon_output0 = self.convert_to_8_channels(outputs)
        
        # 打印调试信息

        
        # 确保所有输出尺寸一致
        assert all(out.shape[2:] == input_size for out in [
            bicon_output0, bicon_output1, bicon_output2, 
            bicon_output3, bicon_output4, main_output
        ]), "Output sizes do not match"
        
        # Concatenate all outputs
        bicon_outputs = torch.cat([
            bicon_output0, bicon_output1, bicon_output2, 
            bicon_output3, bicon_output4
        ], dim=1)
        
        outputs_to_return = torch.cat([
            bicon_outputs, main_output, 
            output4, output3, output2, output1
        ], dim=1)
        
        # print(f"Final output shape: {outputs_to_return.shape}")
        
        return outputs_to_return

    def to(self, device):
        """Ensures all submodules are moved to the specified device"""
        super().to(device)
        for conv in self.alpha_convs:
            conv.to(device)
        for conv in self.beta_convs:
            conv.to(device)
        self.initial_conv.to(device)
        return self

if __name__ == '__main__':
    # 测试代码
    model = RetiFluidNet(in_channels=1, num_classes=4)
    x = torch.randn(4, 1, 256, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
