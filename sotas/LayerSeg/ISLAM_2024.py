"""
@inproceedings{islam2024uncertainty,
  title={Uncertainty-aware retinal layer segmentation in OCT through probabilistic signed distance functions},
  author={Islam, Mohammad Mohaiminul and de Vente, Coen and Liefers, Bart and Klaver, Caroline and Bekkers, Erik J and S{\'a}nchez, Clara I},
  booktitle={Medical Imaging with Deep Learning},
  year={2024}
}
"""

import torch
import torch.nn as nn

class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x

class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ASPP(nn.Module):

    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18],group=1,groupNorm=False):
        super().__init__()
    
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c,
                      out_c,
                      kernel_size=3,
                      dilation=rate[0],
                      padding=rate[0],groups=group),
                      nn.GroupNorm(num_groups=group,num_channels=out_c) 
                      if groupNorm else nn.BatchNorm2d(out_c))

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c,
                      out_c,
                      kernel_size=3,
                      dilation=rate[1],
                      padding=rate[1],groups=group), 
                      nn.GroupNorm(num_groups=group,num_channels=out_c) 
                      if groupNorm else nn.BatchNorm2d(out_c))


        self.c3 = nn.Sequential(
            nn.Conv2d(in_c,
                      out_c,
                      kernel_size=3,
                      dilation=rate[2],
                      padding=rate[2],groups=group), 
                      nn.GroupNorm(num_groups=group,num_channels=out_c) 
                      if groupNorm else nn.BatchNorm2d(out_c))

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c,
                      out_c,
                      kernel_size=3,
                      dilation=rate[3],
                      padding=rate[3],groups=group), 
                      nn.GroupNorm(num_groups=group,num_channels=out_c) 
                      if groupNorm else nn.BatchNorm2d(out_c))

        self.c5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0,groups=group)

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.c5(x)
        return y

class Attention_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(),
            nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2))
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1),
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y

class CustomHead(nn.Module):

    def __init__(self, num_class=1,activation=False) -> None:
        super().__init__()
        self.activation=activation
        self.d6 = Decoder_Block([16, 64], 32)
        self.aspp = ASPP(32, 8)
        self.output =nn.Sequential(
            nn.Conv2d(8, num_class,kernel_size=1, padding=0)
            )

    def forward(self, c1, d5):
        
        d6 = self.d6(c1, d5)
        out = self.aspp(d6)
        out = self.output(out)
        if self.activation:
            out = nn.functional.relu(out)
        return out




class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a1 = Attention_Block(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNet_Block(in_c[0]+in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d

class ISLAM(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=3,
                 gaussian_output=False,
                 out_act=False,
                 groupNorm=False,
                 use_multi_head=False,
                 use_input_instance_norm=True) -> None:
        """
        Args:
            in_channels (int): Number of input channels (default: 1)
            num_classes (int): Number of output classes (default: 3)
            gaussian_output (bool): Whether to output Gaussian parameters (default: False)
            out_act (bool): Whether to use activation on output (default: False)
            groupNorm (bool): Whether to use group normalization (default: False)
            use_multi_head (bool): Whether to use multiple output heads (default: False)
            use_input_instance_norm (bool): Whether to normalize input (default: True)
        """
        super().__init__()

        self.gaussian_output = gaussian_output
        self.use_multi_head = use_multi_head
        self.in_channels = in_channels
        self.use_input_instance_norm = use_input_instance_norm
        self.out_act = out_act

        # Encoder path
        self.c1 = Stem_Block(in_channels, 16, stride=1)
        self.c2 = ResNet_Block(16, 32, stride=2)
        self.c3 = ResNet_Block(32, 64, stride=2)
        self.c4 = ResNet_Block(64, 128, stride=2)
        self.c5 = ResNet_Block(128, 256, stride=2)
        self.c6 = ResNet_Block(256, 512, stride=2)

        # Bottleneck
        self.b1 = ASPP(512, 1024)

        # Decoder path
        self.d2 = Decoder_Block([256, 1024], 512)
        self.d3 = Decoder_Block([128, 512], 256)
        self.d4 = Decoder_Block([64, 256], 128)
        self.d5 = Decoder_Block([32, 128], 64)
     
        if self.use_multi_head:
            # Multi-head output
            self.l1 = CustomHead(num_class=1, activation=self.out_act)
            self.l2 = CustomHead(num_class=1, activation=self.out_act)
            self.l3 = CustomHead(num_class=1, activation=self.out_act)
            
            if self.gaussian_output:
                self.log_var_1 = CustomHead(num_class=1, activation=True)
                self.log_var_2 = CustomHead(num_class=1, activation=True)
                self.log_var_3 = CustomHead(num_class=1, activation=True)
        else:
            # Single head output
            self.d5 = Decoder_Block([32, 128], 81)
            self.d6 = Decoder_Block([16, 81], 81)
            self.aspp = ASPP(81, 27, group=3, groupNorm=groupNorm)
            
            if groupNorm:
                self.output = nn.Sequential(
                    nn.Conv2d(27, 9, groups=3, kernel_size=1, padding=0),
                    nn.GroupNorm(num_groups=3, num_channels=9),
                    nn.Conv2d(9, num_classes, kernel_size=1, padding=0)
                )
            else:
                self.output = nn.Sequential(
                    nn.Conv2d(27, 9, groups=3, kernel_size=1, padding=0),
                    nn.Conv2d(9, num_classes, kernel_size=1, padding=0)
                )

    def forward(self, inputs):
        if self.use_input_instance_norm:
            inputs=nn.InstanceNorm2d(self.in_channels)(inputs)
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        c6 = self.c6(c5)

        b1 = self.b1(c6)

        d2 = self.d2(c5, b1)
        d3 = self.d3(c4, d2)
        d4 = self.d4(c3, d3)
        d5 = self.d5(c2, d4)

        if self.use_multi_head:
            l1=self.l1(c1,d5)
            l2=self.l2(c1,d5)
            l3=self.l3(c1,d5)
            
            if self.gaussian_output:
                log_var_1=self.log_var_1(c1,d5)
                log_var_2=self.log_var_2(c1,d5)
                log_var_3=self.log_var_3(c1,d5)
                return (torch.cat([l1,l2,l3],dim=1),torch.cat([log_var_1,log_var_2,log_var_3],dim=1))
            else:
                return torch.cat([l1,l2,l3],dim=1)
        else:
            d5 = self.d5(c2, d4)
            d6 = self.d6(c1, d5)
            output = self.aspp(d6)
            output =self.output(output)
            return output



# from torchinfo import summary

if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型实例
    model = ISLAM(
        in_channels=1,
        num_classes=4,
        gaussian_output=False,
        use_multi_head=False
    ).to(device)
    
    # 创建测试输入
    x = torch.randn(4, 1, 512, 512).to(device)
    
    # 测试前向传播
    try:
        with torch.no_grad():
            out = model(x)
            print(f"\nInput shape: {x.shape}")
            print(f"Output shape: {out.shape}")
            
            # 打印模型参数数量
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params:,}")
            
    except Exception as e:
        print(f"\nError during forward pass: {str(e)}")
        import traceback
        traceback.print_exc()