import torch
from torch import nn, einsum
import math
import warnings
from timm.models.layers import DropPath
from torch.nn.modules.utils import _pair as to_2tuple
# from mmcv.runner import BaseModule
# from mmcv.cnn.utils.weight_init import (constant_init, normal_init,trunc_normal_init)
from einops.layers.torch import Rearrange
from einops import rearrange,repeat
from timm.models.layers import trunc_normal_

"""
@article{he2024light,
  title={Light-weight Retinal Layer Segmentation with Global Reasoning},
  author={He, Xiang and Song, Weiye and Wang, Yiming and Poiesi, Fabio and Yi, Ji and Desai, Manishi and Xu, Quanqing and Yang, Kongzheng and Wan, Yi},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2024},
  publisher={IEEE}
}
"""

# X. He et al., "Lightweight Retinal Layer Segmentation With Global Reasoning," 
# in IEEE Transactions on Instrumentation and Measurement, 
# vol. 73, pp. 1-14, 2024, Art no. 2520214, doi: 10.1109/TIM.2024.3400305.
#https://github.com/Medical-Image-Analysis/LightReSeg/blob/main/README.md

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.apply(self._init_weights)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def contracting_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels,padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(kernel_size=(3, 3), in_channels=out_channels, out_channels=out_channels,padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )
    return block

class expansive_block(nn.Module):
    def __init__(self, in_channels):
        super(expansive_block, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=(3, 3), stride=2, padding=1,
                                     output_padding=1, dilation=1)  # in_channels//2,无论除数是整数还是小数，向下取整
        self.attention_module=AttentionModule(in_channels//2)
        self._init_weight()
    def forward(self, e, d):
        d = self.up(d)
        e=0.8*self.attention_module(e)+e
        # cat = torch.cat([e, d], dim=1)
        out = e+d
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(1, 1), in_channels=in_channels, out_channels=out_channels),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
    )
    return block

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes,stride,padding,bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=stride, padding=padding,dilation=1, groups=inplanes, bias=bias)
        self.pointwise1 = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, dilation=1,groups=planes, bias=bias)
        self.pointwise2 = nn.Conv2d(planes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.pointwise2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ChannelAttentionModule(nn.Module):

    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, C, H, W = x.shape

        #  N*C*(H*W)
        query = x.view(N, C, -1)

        #  N*(H*W)*C
        key = x.view(N, C, -1).permute(0, 2, 1)

        # N * C * C
        energy = torch.bmm(query, key)
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        attention = self.softmax(energy)

        value = x.view(N, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(N, C, H, W)
        out = self.gamma * out + x

        return out

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super(AttentionModule,self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        self.conv3 = nn.Conv2d(4*dim, dim, 1)
        self.channelA = ChannelAttentionModule()
        self.channelB = ChannelAttentionModule()
        self.channelC = ChannelAttentionModule()
        self.channelD = ChannelAttentionModule()

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = torch.cat((self.channelA(attn),self.channelB(attn_0),self.channelC(attn_1),self.channelD(attn_2)),dim=1)
        attn = self.conv3(attn)

        return attn * u

class LightReSeg(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LightReSeg, self).__init__()
        num_class=num_classes
        in_channel=in_channels
        # Encode
        self.conv_encode1 = contracting_block(in_channels=in_channel, out_channels=16)
        self.conv_pool1=SeparableConv2d(16,16,stride=2,padding=1)

        self.conv_encode2 = contracting_block(in_channels=16, out_channels=32)
        self.conv_pool2=SeparableConv2d(32,32,stride=2,padding=1)

        self.conv_encode3 = contracting_block(in_channels=32, out_channels=64)
        self.conv_pool3=SeparableConv2d(64,64,stride=2,padding=1)

        self.conv_encode4 = contracting_block(in_channels=64, out_channels=128)
        self.conv_pool4=SeparableConv2d(128,128,stride=2,padding=1)

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=128, out_channels=256,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=256,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )


        # Decode
        self.conv_decode4 = expansive_block(256)
        self.conv_decode3 = expansive_block(128)
        self.conv_decode2 = expansive_block(64)
        self.conv_decode1 = expansive_block(32)
        self.final_layer = final_block(16, num_class)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
            nn.Linear(128, 128))
        # self.recover =nn.Sequential(
        #     Rearrange(' b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=1, p2=1,h=14),) #Change the value of h based on the size of the input image
        self.recover =nn.Sequential(
            Rearrange(' b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=1, p2=1,h=32),) #Change the value of h based on the size of the input image
        self.pos_embedding = nn.Parameter(torch.randn(1, 1445, 128))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        self.dropout = nn.Dropout()
        self.transformer = Transformer(dim=128, depth=3, heads=8, dim_head=64, mlp_dim=768, dropout=0.)

    def forward(self, x):

        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_pool1(encode_block1)

        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_pool2(encode_block2)

        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_pool3(encode_block3)

        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_pool4(encode_block4)

        x = self.to_patch_embedding(encode_pool4)
        b, n, _ = x.shape
        cls_tokens=repeat(self.cls_token,'() n d -> b n d',b=b)
        x=torch.cat((cls_tokens,x),dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)[:,1:,:]
        encode_pool4=self.recover(x)+encode_pool4
        bottleneck = self.bottleneck(encode_pool4)

        decode_block4 = self.conv_decode4(encode_block4, bottleneck)
        decode_block3 = self.conv_decode3(encode_block3, decode_block4)
        decode_block2 = self.conv_decode2(encode_block2, decode_block3)
        decode_block1 = self.conv_decode1(encode_block1, decode_block2)

        final_layer = self.final_layer(decode_block1)
        return final_layer


if __name__ == "__main__":
    import torch as t
    # rgb = t.randn(2, 1, 224, 224)
    rgb = t.randn(2, 1, 512, 512)
    net = LightReSeg(in_channels=1,num_classes=7)
    out = net(rgb)
    print(out.shape)