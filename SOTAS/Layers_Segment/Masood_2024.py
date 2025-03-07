"""@article{masood2024deep,
  title={Deep choroid layer segmentation using hybrid features extraction from OCT images},
  author={Masood, Saleha and Ali, Saba Ghazanfar and Wang, Xiangning and Masood, Afifa and Li, Ping and Li, Huating and Jung, Younhyun and Sheng, Bin and Kim, Jinman},
  journal={The Visual Computer},
  volume={40},
  number={4},
  pages={2775--2792},
  year={2024},
  publisher={Springer}
}"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage

class GaborFeatures(nn.Module):
    def __init__(self):
        super(GaborFeatures, self).__init__()
        # 根据论文使用更多的方向和频率
        self.orientations = [0, 45, 90, 135, -45, -135]  # 6个方向
        self.frequencies = [0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 8个频率
        self.sigma = 1.0
        self.output_dim = len(self.orientations) * len(self.frequencies) * 64  # 3072维特征向量
        
    def forward(self, x):
        gabor_features = []
        for theta in self.orientations:
            for freq in self.frequencies:
                kernel = self._gabor_kernel(freq, theta)
                kernel_tensor = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
                if x.is_cuda:
                    kernel_tensor = kernel_tensor.cuda()
                feature = F.conv2d(x, kernel_tensor, padding='same')
                gabor_features.append(feature)
        return torch.cat(gabor_features, dim=1)
    
    def _gabor_kernel(self, frequency, theta):
        theta = theta / 180.0 * np.pi
        kernel_size = int(2 * np.ceil(2.5 * self.sigma) + 1)
        y, x = np.mgrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        gb = np.exp(-.5 * (x_theta**2 + y_theta**2) / self.sigma**2) * np.cos(2 * np.pi * frequency * x_theta)
        return gb

class HaarFeatures(nn.Module):
    def __init__(self):
        super(HaarFeatures, self).__init__()
        self.haar_kernels = self._create_haar_kernels()
        
    def forward(self, x):
        haar_features = []
        for kernel in self.haar_kernels:
            kernel_tensor = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
            if x.is_cuda:
                kernel_tensor = kernel_tensor.cuda()
            feature = F.conv2d(x, kernel_tensor, padding='same')
            haar_features.append(feature)
        return torch.cat(haar_features, dim=1)
    
    def _create_haar_kernels(self):
        kernels = []
        # Horizontal
        kernels.append(np.array([[1, 1], [-1, -1]]))
        # Vertical
        kernels.append(np.array([[1, -1], [1, -1]]))
        # Diagonal
        kernels.append(np.array([[1, -1], [-1, 1]]))
        return kernels

class GLCMFeatures(nn.Module):
    def __init__(self):
        super(GLCMFeatures, self).__init__()
        # 根据论文使用4个方向
        self.angles = [0, 90, -45, -135]  # 水平、垂直和两个对角方向
        self.distances = [1, 2]  # 使用多个距离
        # 使用skimage支持的属性
        self.props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 
                     'correlation', 'ASM']  # 基本属性
        
    def _preprocess_image(self, img):
        """将浮点数图像转换为uint8类型"""
        # 确保值在[0,1]范围内
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        # 转换到[0,255]范围
        img = (img * 255).astype(np.uint8)
        return img
    
    def _calculate_entropy(self, glcm):
        """手动计算GLCM熵"""
        eps = 1e-8
        glcm = glcm + eps  # 避免log(0)
        entropy = -np.sum(glcm * np.log2(glcm))
        return entropy
    
    def _calculate_variance(self, glcm):
        """手动计算GLCM方差"""
        rows, cols = glcm.shape[:2]
        i, j = np.ogrid[0:rows, 0:cols]
        mean = np.sum(i * glcm)
        variance = np.sum(((i - mean) ** 2) * glcm)
        return variance
        
    def forward(self, x):
        x_np = x.cpu().numpy()
        batch_features = []
        
        for i in range(x_np.shape[0]):
            img = x_np[i, 0]
            # 预处理图像
            img = self._preprocess_image(img)
            
            features = []
            for angle in self.angles:
                for d in self.distances:
                    try:
                        glcm = graycomatrix(img, [d], [angle], 
                                          levels=256,  # 指定灰度级别
                                          symmetric=True, 
                                          normed=True)
                        
                        # 计算基本属性
                        for prop in self.props:
                            feature = graycoprops(glcm, prop)
                            features.extend(feature.flatten())
                            
                        # 添加手动计算的属性
                        entropy = self._calculate_entropy(glcm[:, :, 0, 0])
                        variance = self._calculate_variance(glcm[:, :, 0, 0])
                        features.extend([entropy, variance])
                        
                    except Exception as e:
                        print(f"Error in GLCM calculation: {str(e)}")
                        print(f"Image stats - min: {img.min()}, max: {img.max()}, dtype: {img.dtype}")
                        raise e
                        
            batch_features.append(features)
            
        glcm_features = torch.FloatTensor(batch_features).unsqueeze(-1).unsqueeze(-1)
        if x.is_cuda:
            glcm_features = glcm_features.cuda()
        return glcm_features.expand(-1, -1, x.size(2), x.size(3))

class CNNBranch(nn.Module):
    """5层卷积+3层全连接的CNN分支"""
    def __init__(self, in_channels):
        super(CNNBranch, self).__init__()
        # 5层卷积层，最后输出64通道
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 128, 3, stride=1, padding=1)  # 减少通道数
        self.conv5 = nn.Conv2d(128, 64, 3, stride=1, padding=1)   # 最终输出64通道
        
        # Batch Normalization层
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)  # 对应修改
        self.bn5 = nn.BatchNorm2d(64)   # 对应修改
        
        # 3层全连接层 - 可选
        self.use_fc = False  # 默认不使用全连接层
        if self.use_fc:
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 64)
        
    def forward(self, x):
        # 保存输入尺寸
        input_size = x.shape[2:]
        
        # 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # 上采样回原始尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x

class Masood2024(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(Masood2024, self).__init__()
        
        # CNN branches
        self.cnn_branch1 = CNNBranch(in_channels)
        self.cnn_branch2 = CNNBranch(in_channels)
        self.cnn_branch3 = CNNBranch(in_channels)
        self.cnn_branch4 = CNNBranch(in_channels)
        
        # Handcrafted feature extractors
        self.gabor = GaborFeatures()
        self.haar = HaarFeatures()
        self.glcm = GLCMFeatures()
        
        # Feature fusion
        total_features = 256 + 48 + 3 + 64  # 64*4 + 48(Gabor) + 3(Haar) + 8(GLCM)
        
        self.final_conv = nn.Conv2d(total_features, num_classes, 1)
        
    def forward(self, x):
        # 保存输入尺寸用于检查
        input_size = x.shape[2:]
        
        # CNN features
        cnn1 = self.cnn_branch1(x)
        cnn2 = self.cnn_branch2(x)
        cnn3 = self.cnn_branch3(x)
        cnn4 = self.cnn_branch4(x)
        
        # Handcrafted features
        gabor_feat = self.gabor(x)
        haar_feat = self.haar(x)
        glcm_feat = self.glcm(x)
        
        # 打印调试信息
        # print(f"CNN1 shape: {cnn1.shape}")
        # print(f"Gabor shape: {gabor_feat.shape}")
        # print(f"Haar shape: {haar_feat.shape}")
        # print(f"GLCM shape: {glcm_feat.shape}")
        
        # 确保所有特征具有相同的空间维度
        # gabor_feat = F.interpolate(gabor_feat, size=input_size, mode='bilinear', align_corners=True)
        # haar_feat = F.interpolate(haar_feat, size=input_size, mode='bilinear', align_corners=True)
        # glcm_feat = F.interpolate(glcm_feat, size=input_size, mode='bilinear', align_corners=True)
        
        # Concatenate all features
        combined = torch.cat([cnn1, cnn2, cnn3, cnn4, gabor_feat, haar_feat, glcm_feat], dim=1)
        
        # Final classification
        out = self.final_conv(combined)
        return torch.sigmoid(out)

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # 创建更合理的测试输入
        batch_size = 2
        in_channels = 1
        img_size = 512
        num_classes = 2
        
        # 生成[0,1]范围内的随机数据
        x = torch.rand(batch_size, in_channels, img_size, img_size).to(device)
        
        # 标准化
        x = (x - 0.5) / 0.5
        
        # 创建模型实例
        model = Masood2024(
            in_channels=in_channels,
            num_classes=num_classes
        ).to(device)
        
        # 测试前向传播
        with torch.no_grad():
            outputs = model(x)
            
            print("\nFeature dimensions:")
            print(f"Input shape: {x.shape}")
            print(f"Input range: [{x.min().item():.3f}, {x.max().item():.3f}]")
            
            # CNN分支输出
            cnn_out = model.cnn_branch1(x)
            print(f"CNN branch output shape: {cnn_out.shape}")
            
            # Gabor特征输出
            gabor_out = model.gabor(x)
            print(f"Gabor features shape: {gabor_out.shape}")
            print(f"Gabor feature dimension: {model.gabor.output_dim}")
            
            # Haar特征输出
            haar_out = model.haar(x)
            print(f"Haar features shape: {haar_out.shape}")
            
            # GLCM特征输出
            glcm_out = model.glcm(x)
            print(f"GLCM features shape: {glcm_out.shape}")
            
            # 最终输出
            print(f"Final output shape: {outputs.shape}")
            print(f"Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
            
            # 打印模型参数数量
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nTotal parameters: {total_params:,}")
            
            # 计算FLOPs
            from thop import profile
            flops, _ = profile(model, inputs=(x,))
            print(f"FLOPs: {flops/1e9:.2f}G")
            
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()

"""
训练注意事项：

1. 数据预处理：
- 图像需要归一化到[0,1]范围
- 建议使用标准化(mean=0.5, std=0.5)
- 保持图像大小一致(建议512x512)

2. 特征提取器相关：
- Gabor特征计算较慢，建议使用多进程预处理
- GLCM特征需要将图像量化到较少的灰度级(如256级)
- 手工特征和CNN特征的尺度可能不一致，需要注意特征融合

3. 训练策略：
- 使用较小的学习率(如1e-4)
- 建议使用Adam优化器
- 使用学习率衰减策略
- 考虑使用梯度裁剪防止梯度爆炸

4. 内存管理：
- 由于特征维度较大，建议使用较小的batch size
- 必要时使用梯度累积
- 考虑使用混合精度训练

5. 验证和测试：
- 使用Dice系数和IoU评估分割效果
- 保存最佳验证性能的模型
- 测试时使用TTA(Test Time Augmentation)

6. 其他注意事项：
- 确保GLCM计算时CPU和GPU之间的数据传输效率
- 监控各类特征的数值范围，确保特征融合的有效性
- 可以考虑使用特征选择方法减少冗余特征
"""
