import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import SegformerForSemanticSegmentation

"""
@article{yang2024anomaly,
  title={Anomaly-guided weakly supervised lesion segmentation on retinal OCT images},
  author={Yang, Jiaqi and Mehta, Nitish and Demirci, Gozde and Hu, Xiaoling and Ramakrishnan, Meera S and Naguib, Mina and Chen, Chao and Tsai, Chia-Ling},
  journal={Medical Image Analysis},
  volume={94},
  pages={103139},
  year={2024},
  publisher={Elsevier}
}
"""

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=640, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SegformerEncoder(nn.Module):
    def __init__(self, image_size=640, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

class SegformerDecodeHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        self.linear_c = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        
        x = self.linear_c(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.dropout(x)
        x = self.linear_pred(x)
        
        return x

class SegformerForSemanticSegmentation(nn.Module):
    def __init__(self, image_size=640, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12, num_classes=3):
        super().__init__()
        self.encoder = SegformerEncoder(image_size, patch_size, in_channels, embed_dim, depth, num_heads)
        self.decode_head = SegformerDecodeHead(embed_dim, num_classes)
        
        # 配置信息
        self.config = {
            'image_size': image_size,
            'patch_size': patch_size,
            'in_channels': in_channels,
            'embed_dim': embed_dim,
            'depth': depth,
            'num_heads': num_heads,
            'num_classes': num_classes
        }
        
    def forward(self, pixel_values):
        encoder_outputs = self.encoder(pixel_values)
        logits = self.decode_head(encoder_outputs)
        
        # 上采样到原始图像大小
        logits = F.interpolate(logits, size=(self.config['image_size'], self.config['image_size']), 
                             mode='bilinear', align_corners=False)
        
        return {"logits": logits}

    @classmethod
    def from_pretrained(cls, model_name):
        # 这里简化处理，根据model_name返回预定义的配置
        if 'b5' in model_name:
            return cls(
                image_size=640,
                patch_size=16,
                in_channels=3,
                embed_dim=768,
                depth=12,
                num_heads=12,
                num_classes=3
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")


class SegFormerModelWeakly(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(SegFormerModelWeakly, self).__init__()
        model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
        self.model = SegformerForSemanticSegmentation(
            image_size=640,
            patch_size=16,
            in_channels=in_channels,
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_classes=num_classes
        )
        
        # 不再需要修改classifier，因为已经在初始化时设置了正确的类别数
        # 不再需要segformer属性
        self.config = self.model.config
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, pixel_values):
        outputs = self.model(pixel_values)["logits"]
        cls_res = self.avgpool(outputs)
        cls_res = cls_res.view(cls_res.size(0), -1)
        return cls_res, outputs


class SegFormerModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(SegFormerModel, self).__init__()
        model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
        self.model = SegformerForSemanticSegmentation(
            image_size=640,
            patch_size=16,
            in_channels=in_channels,
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_classes=num_classes
        )
        
        # 不再需要修改classifier和segformer属性
        self.config = self.model.config
    
    def forward(self, pixel_values):
        outputs = self.model(pixel_values)["logits"]
        outputs = F.interpolate(outputs, size=(640, 640), mode="bilinear", align_corners=False)
        return outputs


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_classes, n_channels=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint.checkpoint(self.outc)


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # # 测试UNet
    # unet = UNet(n_classes=3, n_channels=3).to(device)
    x = torch.randn(4, 3, 640, 640).to(device)
    
   
    segformer = SegFormerModel(in_channels=3, num_classes=3).to(device)
    try:
        with torch.no_grad():
            out = segformer(x)
            print(f"\nSegFormer test:")
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {out.shape}")
    except Exception as e:
        print(f"\nError during SegFormer forward pass: {str(e)}")
        import traceback
        traceback.print_exc() 