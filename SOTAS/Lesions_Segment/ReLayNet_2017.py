"""
@article{roy2017relaynet,
  title={ReLayNet: retinal layer and fluid segmentation of macular optical coherence tomography using fully convolutional networks},
  author={Roy, Abhijit Guha and Conjeti, Sailesh and Karri, Sri Phani Krishna and Sheet, Debdoot and Katouzian, Amin and Wachinger, Christian and Navab, Nassir},
  journal={Biomedical optics express},
  volume={8},
  number={8},
  pages={3627--3642},
  year={2017},
  publisher={Optica Publishing Group}
}
"""

"""ClassificationCNN"""
import torch
import torch.nn as nn




class ReLayNet(nn.Module):
    """
    A PyTorch implementation of ReLayNet
    
    Args:
        in_channels (int): Number of input channels (default: 1)
        num_classes (int): Number of output classes (default: 10)
        num_filters (int): Number of filters in first conv layer (default: 64)
        kernel_h (int): Kernel height (default: 7)
        kernel_w (int): Kernel width (default: 3)
        stride_conv (int): Stride for convolution (default: 1)
        pool (int): Pooling size (default: 2)
        stride_pool (int): Stride for pooling (default: 2)
    """

    def __init__(self, in_channels=1, num_classes=10, num_filters=64, kernel_h=7, kernel_w=3,
                 stride_conv=1, pool=2, stride_pool=2):
        super(ReLayNet, self).__init__()
        
        # 构建基础参数字典
        base_params = {
            'num_channels': in_channels,
            'num_filters': num_filters,
            'kernel_h': kernel_h,
            'kernel_w': kernel_w,
            'stride_conv': stride_conv,
            'pool': pool,
            'stride_pool': stride_pool,
            'kernel_c': 1
        }

        # Encoder path
        self.encode1 = EncoderBlock(base_params)
        
        encoder2_params = base_params.copy()
        encoder2_params['num_channels'] = num_filters
        self.encode2 = EncoderBlock(encoder2_params)
        
        encoder3_params = base_params.copy()
        encoder3_params['num_channels'] = num_filters
        self.encode3 = EncoderBlock(encoder3_params)
        
        # Bottleneck
        bottleneck_params = base_params.copy()
        bottleneck_params['num_channels'] = num_filters
        self.bottleneck = BasicBlock(bottleneck_params)
        
        # Decoder path
        # 解码器的输入通道数需要考虑skip connection的拼接
        decoder1_params = base_params.copy()
        decoder1_params['num_channels'] = num_filters * 2  # bottleneck + skip
        decoder1_params['num_filters'] = num_filters
        self.decode1 = DecoderBlock(decoder1_params)
        
        decoder2_params = base_params.copy()
        decoder2_params['num_channels'] = num_filters * 2
        decoder2_params['num_filters'] = num_filters
        self.decode2 = DecoderBlock(decoder2_params)
        
        decoder3_params = base_params.copy()
        decoder3_params['num_channels'] = num_filters * 2
        decoder3_params['num_filters'] = num_filters
        self.decode3 = DecoderBlock(decoder3_params)
        
        # Classifier
        classifier_params = base_params.copy()
        classifier_params['num_channels'] = num_filters
        classifier_params['num_class'] = num_classes
        self.classifier = ClassifierBlock(classifier_params)

    def forward(self, input):
        # Encoder path
        e1, out1, ind1 = self.encode1.forward(input)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)
        
        # Bottleneck
        bn = self.bottleneck.forward(e3)
        
        # Decoder path with skip connections
        d3 = self.decode1.forward(bn, out3, ind3)
        d2 = self.decode2.forward(d3, out2, ind2)
        d1 = self.decode3.forward(d2, out1, ind1)
        
        # Classification
        prob = self.classifier.forward(d1)

        return prob

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

# List of APIs
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    '''
    param ={
        'num_channels':1,
        'num_filters':64,
        'kernel_h':7,
        'kernel_w':3,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':10
    }

    '''

    def __init__(self, params):
        super(BasicBlock, self).__init__()

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        self.conv = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                              kernel_size=(params['kernel_h'], params['kernel_w']),
                              padding=(padding_h, padding_w),
                              stride=params['stride_conv'])
        self.batchnorm = nn.BatchNorm2d(num_features=params['num_filters'])
        self.prelu = nn.PReLU()

    def forward(self, input):
        out_conv = self.conv(input)
        out_bn = self.batchnorm(out_conv)
        out_prelu = self.prelu(out_bn)
        return out_prelu


class EncoderBlock(BasicBlock):
    def __init__(self, params):
        super(EncoderBlock, self).__init__(params)
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input):
        out_block = super(EncoderBlock, self).forward(input)
        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


class DecoderBlock(BasicBlock):
    def __init__(self, params):
        super(DecoderBlock, self).__init__(params)
        self.unpool = nn.MaxUnpool2d(kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, input, out_block, indices):
        unpool = self.unpool(input, indices)
        concat = torch.cat((out_block, unpool), dim=1)
        out_block = super(DecoderBlock, self).forward(concat)

        return out_block


class ClassifierBlock(nn.Module):
    def __init__(self, params):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(params['num_channels'], params['num_class'], params['kernel_c'], params['stride_conv'])
        self.softmax = nn.Softmax2d()

    def forward(self, input):
        out_conv = self.conv(input)
        #out_logit = self.softmax(out_conv)
        return out_conv
    
if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型实例
    model = ReLayNet(
        in_channels=1,
        num_classes=4,       
    ).to(device)
    
    # 创建测试输入
    x = torch.randn(4, 1, 256, 256).to(device)
    
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
