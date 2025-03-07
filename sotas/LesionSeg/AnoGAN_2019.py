import torch
import torch.nn as nn
import torch.nn.functional as F
"""
@article{schlegl2019f,
  title={f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks},
  author={Schlegl, Thomas and Seeb{\"o}ck, Philipp and Waldstein, Sebastian M and Langs, Georg and Schmidt-Erfurth, Ursula},
  journal={Medical image analysis},
  volume={54},
  pages={30--44},
  year={2019},
  publisher={Elsevier}
}
"""
##
class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, bias=False)
        )

    def forward(self, x):
        return self.model(x)


##
class Decoder(nn.Module):
    def __init__(self, out_channels=1):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


##
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Generator, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, x, mode='train'):
        if mode == 'train':
            # 训练模式：直接通过编码器和解码器
            features = self.encoder(x)
            output = self.decoder(features)
            return features, output
        else:
            # 测试模式：只通过解码器
            return self.decoder(x)


##
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(in_channels)
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1),
            nn.Conv2d(32, 1, kernel_size=1, stride=1)
        )

    def forward(self, x):
        features = self.encoder(x)
        pred = self.classifier(features)
        return features, pred


##
class AnoGAN(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(AnoGAN, self).__init__()
        self.G = Generator(in_channels, num_classes)
        self.D = Discriminator(in_channels)
        
    def forward(self, x, mode='train'):
        if mode == 'train':
            # 训练模式
            g_features, fake_images = self.G(x)
            d_features_real, d_pred_real = self.D(x)
            d_features_fake, d_pred_fake = self.D(fake_images)
            
            return {
                'g_features': g_features,
                'fake_images': fake_images,
                'd_features_real': d_features_real,
                'd_pred_real': d_pred_real,
                'd_features_fake': d_features_fake,
                'd_pred_fake': d_pred_fake
            }
        else:
            # 测试模式：只使用生成器
            g_features, reconstructed = self.G(x)
            return reconstructed

    def encode(self, x):
        """编码器功能"""
        return self.G.encoder(x)

    def decode(self, z):
        """解码器功能"""
        return self.G.decoder(z)


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型实例
    model = AnoGAN(in_channels=1, num_classes=1).to(device)
    
    # 创建测试输入
    x = torch.randn(4, 1, 256, 256).to(device)
    
    try:
        # 测试训练模式
        with torch.no_grad():
            outputs = model(x, mode='train')
            print("\nTraining mode test:")
            for k, v in outputs.items():
                print(f"{k} shape: {v.shape}")
            
        # 测试推理模式
        with torch.no_grad():
            reconstructed = model(x, mode='test')
            print("\nTest mode test:")
            print(f"Input shape: {x.shape}")
            print(f"Reconstructed shape: {reconstructed.shape}")
            
        # 测试编码器和解码器
        with torch.no_grad():
            latent = model.encode(x)
            decoded = model.decode(latent)
            print("\nEncoder-Decoder test:")
            print(f"Latent shape: {latent.shape}")
            print(f"Decoded shape: {decoded.shape}")
            
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params:,}")
            
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()