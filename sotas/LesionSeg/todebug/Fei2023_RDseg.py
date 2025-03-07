import torch
import torch.nn as nn
import torch.nn.functional as F
"""
code of Segmentation of retinal detachment and retinoschisis in OCT images based on complementary multi-class segmentation networks
没有调试，仍有bug。

优点：
多分支协同学习
TSP分支专注于分割细节
FSP分支结合分类和分割任务
两个分支的互补性可以提高分割精度
2. 多尺度特征利用
使用深度监督(DS)机制
跳跃连接保留低层特征
TCIP模块捕获上下文信息
3. 注意力机制的应用
TCIP模块中的自适应池化
水平和垂直方向的注意力
有助于捕获长距离依赖
4. 灵活的损失函数设计
分割使用Dice Loss
分类使用CE Loss
DS损失的权重递减策略
缺点：
计算开销大
双编码器结构（TSP和FSP各一个）
多个解码器阶段
训练和推理时间可能较长
参数量大
多个完整的编码器-解码器路径
可能容易过拟合，特别是在小数据集上
内存消耗高
需要存储多个特征图
DS输出需要额外内存
可能限制batch size
调参难度
多个损失函数的权重平衡
各个模块的超参数选择
训练策略的制定
"""

# 在文件开头添加计算特征图尺寸的函数
def calculate_feature_size(input_size, num_pools):
    """计算经过多次下采样后的特征图尺寸"""
    size = input_size
    for _ in range(num_pools):
        size = size // 2
    return size

# Step 1: 定义TCIP模块（Three-dimensional Contextual Information Perception）
class TCIP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TCIP, self).__init__()
        # 三个分支的卷积层
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))  # 1x1 conv
        
        # 将1xH和Wx1的卷积改为自适应池化+1x1卷积的组合
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 在W维度上池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 在H维度上池化
        
        self.conv_h = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Branch 1: 1x1 conv
        b1 = self.sigmoid(self.branch1(x)) * x
        b1 = F.softmax(b1, dim=1)
        
        # Branch 2: H方向注意力
        h_pool = self.pool_h(x)  # 在W维度上池化
        b2 = self.sigmoid(self.conv_h(h_pool))
        b2 = F.interpolate(b2, size=x.shape[2:], mode='bilinear', align_corners=True)
        b2 = b2 * x
        b2 = F.softmax(b2, dim=1)
        
        # Branch 3: W方向注意力
        w_pool = self.pool_w(x)  # 在H维度上池化
        b3 = self.sigmoid(self.conv_w(w_pool))
        b3 = F.interpolate(b3, size=x.shape[2:], mode='bilinear', align_corners=True)
        b3 = b3 * x
        b3 = F.softmax(b3, dim=1)
        
        # 合并三个分支
        output = b1 + b2 + b3 + x
        return output

# Step 2: 定义FMD模块（Feature Map Decoder）
class FMD(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FMD, self).__init__()
        """
        Args:
            in_channels1: 来自classification encoder的通道数
            in_channels2: 来自segmentation encoder的通道数
            out_channels: 输出通道数
        """
        # Conv1 + BN + ReLU for xi1 from classification encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels1, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Conv3 + BN + ReLU for xi2 from segmentation encoder
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # ConvTranspose for TCIP output (high-level features from TCIP)
        self.conv_trans = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        
        # Final Conv3 + BN + ReLU for combining features
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, xi1, xi2, tcip_output):
        # xi1: from classification encoder
        xi1 = self.conv1(xi1)
        
        # xi2: from segmentation encoder
        xi2 = self.conv3(xi2)
        
        # TCIP output - ConvTranspose to upsample
        tcip_output = self.conv_trans(tcip_output)
        
        # Concatenate features
        combined_features = torch.cat([xi1+xi2, tcip_output], dim=1)
        
        # Final conv
        output = self.final_conv(combined_features)
        
        return output

# Step 3: 定义UNetResNetBackbone
class UNetResNetBackbone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetResNetBackbone, self).__init__()
        
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 减少通道数增长倍数
        self.resnet_block1 = self._resnet_block(out_channels * 2, out_channels * 3)
        self.resnet_block2 = self._resnet_block(out_channels * 3, out_channels * 4)
        self.resnet_block3 = self._resnet_block(out_channels * 4, out_channels * 6)
        self.resnet_block4 = self._resnet_block(out_channels * 6, out_channels * 8)
        
        # print(f"Final output channels: {out_channels * 8}")
    
    def _resnet_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        # print(f"\nUNetResNetBackbone forward pass:")
        # print(f"Input shape: {x.shape}")
        
        x1 = self.encoder(x)
        # print(f"Encoder output shape: {x1.shape}")
        
        x2 = self.resnet_block1(x1)
        # print(f"ResBlock1 output shape: {x2.shape}")
        
        x3 = self.resnet_block2(x2)
        # print(f"ResBlock2 output shape: {x3.shape}")
        
        x4 = self.resnet_block3(x3)
        # print(f"ResBlock3 output shape: {x4.shape}")
        
        x5 = self.resnet_block4(x4)
        # print(f"ResBlock4 output shape: {x5.shape}")
        
        return x1, x2, x3, x4, x5

# Step 4: 定义FSPBranch
class FSPBranch(nn.Module):
    def __init__(self, in_channels, branch2_channels, num_classes):
        super(FSPBranch, self).__init__()
        
        # Segmentation Encoder 和 Classification Encoder
        self.segmentation_encoder = UNetResNetBackbone(in_channels, branch2_channels)
        self.classification_encoder = UNetResNetBackbone(in_channels, branch2_channels)
        
        # 计算encoder输出通道数
        encoder_out_channels = branch2_channels * 8
        
        # TCIP模块
        self.tcip = TCIP(encoder_out_channels, encoder_out_channels)
        
        # 分类分支
        self.classification_conv = nn.Conv2d(encoder_out_channels, num_classes, kernel_size=1)
        
        # 解码器 - 确保通道数匹配
        self.decoder = FMD(
            in_channels1=encoder_out_channels,  # 分类编码器输出
            in_channels2=encoder_out_channels,  # 分割编码器输出
            out_channels=encoder_out_channels   # 保持通道数一致
        )
        
        # 最终的输出层
        self.final_conv = nn.Conv2d(encoder_out_channels, num_classes, kernel_size=3, padding=1)
    
    def forward(self, x):
        # 编码器输出
        x1_seg, x2_seg, x3_seg, x4_seg, x5_seg = self.segmentation_encoder(x)
        x1_class, x2_class, x3_class, x4_class, x5_class = self.classification_encoder(x)
        
        # TCIP模块
        SF5 = self.tcip(x5_seg)
        
        # 分类输出
        classification_output = self.classification_conv(x5_class)
        
        # 解码器
        decoder_output = self.decoder(x4_class, x4_seg, SF5+x5_class)
        
        # 最终输出
        final_output = self.final_conv(decoder_output)
        
        return final_output, classification_output
    

# 添加ModifiedDSModule定义
class ModifiedDSModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=3, scale_factor=2):
        super(ModifiedDSModule, self).__init__()
        self.conv_trans = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 添加一个中间卷积层保持特征通道数，用于跳跃连接
        self.conv_skip = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 最后再降低到分类通道数
        self.conv_final = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, kernel_size=1)
        )
        
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, 
            mode='bilinear', 
            align_corners=True
        )
        
    def forward(self, x):
        # 1. 反卷积上采样
        x = self.conv_trans(x)
        x = self.bn_relu(x)
        
        # 2. 保持通道数的特征图，用于跳跃连接
        skip_features = self.conv_skip(x)
        
        # 3. 降低通道数到num_classes
        ds_output = self.conv_final(x)
        
        # 4. 上采样到原始尺寸
        ds_output = self.upsample(ds_output)
        
        return ds_output, skip_features

# 添加DS损失计算函数
def calculate_ds_loss(decoder_outputs, ground_truth, criterion):
    """
    计算每个解码器阶段的监督损失
    
    Args:
        decoder_outputs: 列表，包含每个解码器阶段的输出 [(B,C,H,W), ...]
        ground_truth: 目标分割图 (B,C,H,W)
        criterion: 损失函数（如CrossEntropyLoss）
    
    Returns:
        total_ds_loss: 所有解码器阶段的加权损失和
    """
    total_ds_loss = 0
    weights = [0.4, 0.3, 0.2, 0.1]  # 不同阶段的权重
    
    for output, weight in zip(decoder_outputs, weights):
        # 确保输出和ground truth尺寸匹配
        if output.shape[2:] != ground_truth.shape[2:]:
            output = F.interpolate(
                output, 
                size=ground_truth.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        # 计算当前阶段的损失并加权
        stage_loss = criterion(output, ground_truth) * weight
        total_ds_loss += stage_loss
    
    return total_ds_loss

# 将TSPBranch移到FeiRDsegNet之前
class TSPBranch(nn.Module):
    def __init__(self, in_channels, num_classes=3, input_size=(16, 16)):
        super(TSPBranch, self).__init__()
        
        # 计算每个stage后的特征图大小
        H, W = input_size
        stage0_size = calculate_feature_size(H, 2)
        stage4_size = calculate_feature_size(stage0_size, 4)
        
        # 确保输入通道数是24的倍数
        c = max(24, (in_channels + 23) // 24 * 24)
        
        # Encoder
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1-4: ResNet blocks
        self.stage1 = self._make_resnet_block(c, c*2)
        self.stage2 = self._make_resnet_block(c*2, c*4)
        self.stage3 = self._make_resnet_block(c*4, c*8)
        self.stage4 = self._make_resnet_block(c*8, c*16)
        
        # Bottleneck: CFGF module
        self.cfgf = CFGF(c*16, H=stage4_size, W=stage4_size)
        
        # Decoder stages
        self.decoder_stage1 = ModifiedDSModule(c*16, c*8, scale_factor=4)
        self.decoder_stage2 = ModifiedDSModule(c*8, c*4, scale_factor=4)
        self.decoder_stage3 = ModifiedDSModule(c*4, c*2, scale_factor=4)
        self.decoder_stage4 = ModifiedDSModule(c*2, c, scale_factor=4)
        
        # Final conv
        self.final_conv = nn.Conv2d(c, num_classes, kernel_size=3, padding=1)
    
    def _make_resnet_block(self, in_channels, out_channels):
        """创建ResNet风格的块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 打印输入尺寸和通道数
        B, C, H, W = x.shape
        # print(f"\nTSPBranch forward pass:")
        # print(f"Input shape: {x.shape}")
        
        # Encoder path
        x0 = self.stage0(x)     
        
        x1 = self.stage1(x0)        
        
        x2 = self.stage2(x1)        
        
        x3 = self.stage3(x2)      
        
        x4 = self.stage4(x3)      
        
        # Bottleneck
        x_cfgf = self.cfgf(x4)
        # print(f"CFGF output shape: {x_cfgf.shape}")
        
        # 解码器路径，保存每个阶段的输出
        decoder_outputs = []
        
        # Decoder stage 1
        d1, skip1 = self.decoder_stage1(x_cfgf)
        x3_up = F.interpolate(x3, size=skip1.shape[2:], mode='bilinear', align_corners=True)
        skip1 = skip1 + x3_up  # 现在通道数匹配
        decoder_outputs.append(d1)
        
        # Decoder stage 2
        d2, skip2 = self.decoder_stage2(skip1)
        x2_up = F.interpolate(x2, size=skip2.shape[2:], mode='bilinear', align_corners=True)
        skip2 = skip2 + x2_up
        decoder_outputs.append(d2)
        
        # Decoder stage 3
        d3, skip3 = self.decoder_stage3(skip2)
        x1_up = F.interpolate(x1, size=skip3.shape[2:], mode='bilinear', align_corners=True)
        skip3 = skip3 + x1_up
        decoder_outputs.append(d3)
        
        # Decoder stage 4
        d4, skip4 = self.decoder_stage4(skip3)
        x0_up = F.interpolate(x0, size=skip4.shape[2:], mode='bilinear', align_corners=True)
        skip4 = skip4 + x0_up
        decoder_outputs.append(d4)
        
        # Final output
        output = self.final_conv(skip4)
        print(f"Final output shape: {output.shape}")
        
        return output, decoder_outputs

# 修改FeiRDsegNet的默认参数
class FeiRDsegNet(nn.Module):
    def __init__(self, in_channels, branch1_channels=64, branch2_channels=None, num_classes=5, input_size=(256, 256)):
        super(FeiRDsegNet, self).__init__()
        
        if branch2_channels is None:
            branch2_channels = num_classes
        
        self.unet_backbone = UNetResNetBackbone(in_channels, branch1_channels)
        backbone_out_channels = branch1_channels * 6
        
        self.tsp_branch = TSPBranch(
            in_channels=backbone_out_channels,
            num_classes=3,
            input_size=(input_size[0]//16, input_size[1]//16)
        )
        
        self.fsp_branch = FSPBranch(
            in_channels=backbone_out_channels,
            branch2_channels=branch2_channels,
            num_classes=num_classes
        )
        
        self.decision_fusion = DecisionFusionLayer(num_classes)
        
    def forward(self, x):
        x1, x2, x3, x4, x5 = self.unet_backbone(x)
        
        # TSP branch - 现在返回主输出和4个DS输出
        tsp_output, tsp_ds_outputs = self.tsp_branch(x4)
        
        # FSP branch
        fsp_segmentation_output, fsp_classification_output = self.fsp_branch(x4)
        
        # 特征图尺寸对齐
        if tsp_output.shape[2:] != x.shape[2:]:
            tsp_output = F.interpolate(tsp_output, size=x.shape[2:], 
                                     mode='bilinear', align_corners=True)
        
        if fsp_segmentation_output.shape[2:] != x.shape[2:]:
            fsp_segmentation_output = F.interpolate(fsp_segmentation_output, size=x.shape[2:], 
                                                  mode='bilinear', align_corners=True)
        
        # 决策融合
        fusion_input = torch.cat([tsp_output, fsp_segmentation_output, x], dim=1)
        final_output = self.decision_fusion(fusion_input)
        
        return {
            'final_output': final_output,            # 最终分割输出
            'tsp_ds_outputs': tsp_ds_outputs,       # TSP的4个DS输出
            'tsp_output': tsp_output,               # TSP主输出
            'fsp_seg_output': fsp_segmentation_output,  # FSP分割输出
            'fsp_cls_output': fsp_classification_output  # FSP分类输出
        }

class RDB(nn.Module):
    def __init__(self, channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = channels + i * growth_rate
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, growth_rate, 3, 1, 1),
                nn.ReLU()
            ))
        
        # 添加调试信息
        self.debug_info = {}
        
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            self.debug_info[f'layer_output_{len(features)}'] = out.shape
            features.append(out)
        return torch.cat(features, 1)

# 检查输入数据维度
def check_input_dimensions(x):
    print(f"Input shape: {x.shape}")
    assert len(x.shape) == 4, "Input should be 4D: [batch_size, channels, height, width]"
    
# 检查特征图尺寸
def verify_feature_maps(feature_maps):
    for i, fm in enumerate(feature_maps):
        print(f"Feature map {i} shape: {fm.shape}")

def test_feirdsegnet():
    # 创建测试数据
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    x = torch.randn(batch_size, channels, height, width)
    
    # 初始化模型
    model = FeiRDsegNet()
    model.set_debug_mode(True)
    
    # 前向传播测试
    try:
        output = model(x)
        
    except Exception as e:
        print(f"Test failed: {str(e)}")

def check_dimensions(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        print(f"Dimension mismatch: {tensor1.shape} vs {tensor2.shape}")
        # 进行必要的度调整
        return F.interpolate(tensor1, size=tensor2.shape[2:])
    return tensor1

# 修改 DecisionFusionLayer
class DecisionFusionLayer(nn.Module):
    def __init__(self, num_classes):
        super(DecisionFusionLayer, self).__init__()
        # 计算输通道数：3(TSP) + num_classes(FSP) + in_channels(原图)
        in_channels = 3 + num_classes + 3  # 假设原图是3通道
        
        # 第一组 conv3: C in_channels-64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 第二组 conv3: C 64-128-64
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 最后一个 conv3: C 64-5
        self.conv3 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

# 修改CFGF模块的实现
class CFGF(nn.Module):
    def __init__(self, in_channels, H, W):
        super(CFGF, self).__init__()
        self.in_channels = in_channels
        self.H = H
        self.W = W
        
        # 使用1x1卷积进行通道压缩
        self.conv_reduce = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        
        # 分别处理H和W方向的注意力
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.conv_w = nn.Sequential(
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 最终的融合卷积
        self.conv_final = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        print(f"CFGF input shape: {x.shape}")
        
        # 通道压缩
        feat = self.conv_reduce(x)  # B, C/8, H, W
        
        # 计算H方向的注意力
        h_avg = torch.mean(feat, dim=3, keepdim=True)  # B, C/8, H, 1
        h_attention = self.conv_h(h_avg)  # B, C, H, 1
        
        # 计算W方向的注意力
        w_avg = torch.mean(feat, dim=2, keepdim=True)  # B, C/8, 1, W
        w_attention = self.conv_w(w_avg)  # B, C, 1, W
        
        # 合并注意力
        attention = h_attention * w_attention
        
        # 应用注意力并添加残差连接
        out = x * attention + x
        out = self.conv_final(out)
        
        print(f"CFGF output shape: {out.shape}")
        return out

# 定义DS (Decoder Stage)模块
class DSModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DSModule, self).__init__()
        self.conv_trans = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1
        )
        self.upsample = nn.Upsample(
            scale_factor=2, 
            mode='bilinear', 
            align_corners=True
        )
      
    
    def forward(self, x):
        x = self.conv_trans(x)
        x = self.bn_relu(x)
        x = self.conv3(x)
        x = self.upsample(x)
        return x
    
    # return total_loss, loss_components
# 测试FSPBranch
if __name__ == "__main__":
    # 1. 基本设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. 创建测试数据
    batch_size = 1
    in_channels = 2
    height, width = 256,256
    num_classes = 3
    
    # 创建随机输入数据
    x = torch.randn(batch_size, in_channels, height, width).to(device)
    print(f"\nInput tensor shape: {x.shape}")
    
    # 3. 始化模型
    try:
        model = FeiRDsegNet(
            in_channels=in_channels,
            branch1_channels=3, 
            branch2_channels=5,# 增大初始通道数
            input_size=(height, width)
        ).to(device)
        
        # 打印每个子模块的参数形状
        print("\nModel parameter shapes:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}")
        
        # 4. 打印模型结构和参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # 5. 模型前向传播测试
        model.eval()
        with torch.no_grad():
            try:
                final_output, (tsp_output, fsp_seg_output, fsp_cls_output) = model(x)
                print("\nForward pass successful!")
                print(f"Final output shape: {final_output.shape}")
                print(f"TSP output shape: {tsp_output.shape}")
                print(f"FSP segmentation output shape: {fsp_seg_output.shape}")
                print(f"FSP classification output shape: {fsp_cls_output.shape}")
                
                # 6. 检查输出维度是否符合预期
                expected_final_shape = (batch_size, num_classes, height, width)  # 最终输出应该和输入图像同尺寸
                expected_tsp_shape = (batch_size, 3, height, width)  # TSP输出3类
                expected_fsp_seg_shape = (batch_size, num_classes, height, width)  # FSP分割输出
                expected_fsp_cls_shape = (batch_size, num_classes, height//32, width//32)  # FSP分类输出
                
                assert final_output.shape == expected_final_shape, \
                    f"Final output shape mismatch. Expected {expected_final_shape}, got {final_output.shape}"
                assert tsp_output.shape == expected_tsp_shape, \
                    f"TSP output shape mismatch. Expected {expected_tsp_shape}, got {tsp_output.shape}"
                assert fsp_seg_output.shape == expected_fsp_seg_shape, \
                    f"FSP segmentation output shape mismatch. Expected {expected_fsp_seg_shape}, got {fsp_seg_output.shape}"
                assert fsp_cls_output.shape == expected_fsp_cls_shape, \
                    f"FSP classification output shape mismatch. Expected {expected_fsp_cls_shape}, got {fsp_cls_output.shape}"
                
                # 7. 检查输出值范围
                print(f"\nOutput stats:")
                for name, tensor in [
                    ("Final output", final_output),
                    ("TSP output", tsp_output),
                    ("FSP segmentation", fsp_seg_output),
                    ("FSP classification", fsp_cls_output)
                ]:
                    print(f"\n{name}:")
                    print(f"Min: {tensor.min():.4f}")
                    print(f"Max: {tensor.max():.4f}")
                    print(f"Mean: {tensor.mean():.4f}")
                    print(f"Shape: {tensor.shape}")
                
                print("\nAll tests passed successfully!")
                
            except Exception as e:
                print(f"\nError during forward pass: {str(e)}")
                print(f"Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
                raise
                
    except Exception as e:
        print(f"\nError initializing model: {str(e)}")
        raise

def dice_loss(pred, target, smooth=1.0):
    """
    计算Dice损失
    pred: (B,C,H,W) - 预测的概率图
    target: (B,C,H,W) - one-hot编码的目标
    """
    pred = F.softmax(pred, dim=1)
    
    # 展平预测值和目标值
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def calculate_total_loss(outputs, targets, cls_targets=None):
    """
    计算总损失
    
    Args:
        outputs: 模型输出的字典
        targets: 分割任务的ground truth
        cls_targets: 分类任务的ground truth (可选)
    """
    # 初始化损失函数
    dice_criterion = dice_loss
    ce_criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    loss_components = {}
    
    # 1. 主分割输出的Dice损失
    final_loss = dice_criterion(outputs['final_output'], targets)
    total_loss += final_loss
    loss_components['final_loss'] = final_loss
    
    # 2. TSP的DS损失
    tsp_ds_loss = 0
    weights = [0.4, 0.3, 0.2, 0.1]  # DS权重
    for ds_output, weight in zip(outputs['tsp_ds_outputs'], weights):
        stage_loss = dice_criterion(ds_output, targets) * weight
        tsp_ds_loss += stage_loss
    total_loss += tsp_ds_loss
    loss_components['tsp_ds_loss'] = tsp_ds_loss
    
    # 3. TSP主输出的Dice损失
    tsp_loss = dice_criterion(outputs['tsp_output'], targets)
    total_loss += tsp_loss
    loss_components['tsp_loss'] = tsp_loss
    
    # 4. FSP分割输出的Dice损失
    fsp_seg_loss = dice_criterion(outputs['fsp_seg_output'], targets)
    total_loss += fsp_seg_loss
    loss_components['fsp_seg_loss'] = fsp_seg_loss
    
    # 5. FSP分类输出的CE损失（如果提供了分类目标）
    if cls_targets is not None:
        fsp_cls_loss = ce_criterion(outputs['fsp_cls_output'], cls_targets)
        total_loss += fsp_cls_loss
        loss_components['fsp_cls_loss'] = fsp_cls_loss
    
    return total_loss, loss_components


