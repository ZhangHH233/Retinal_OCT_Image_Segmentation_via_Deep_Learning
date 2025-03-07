import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out, act = nn.ReLU, drop_rate=0.0, kernel_size = 3):
        super(conv_block,self).__init__()
        self.init_conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=1,padding=kernel_size // 2,bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size,stride=1,padding=kernel_size // 2,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.Dropout2d(drop_rate, inplace=True),
            act(),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size,stride=1,padding=kernel_size // 2,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.Dropout2d(drop_rate, inplace=True),
        )
        self.activation = act()

    def forward(self,x):
        init_conv = self.init_conv(x)
        x = self.conv(init_conv) + init_conv
        x = self.activation(x)
        return x
    

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, act = nn.ReLU, drop_rate=0.0, scale_factor=2):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
            nn.Dropout2d(drop_rate, inplace=True),
			act()
        )

    def forward(self,x):
        x = self.up(x)
        return x
    
class DifferentiableRounding(torch.autograd.Function):
    

    @staticmethod
    def forward(ctx, input):
        return input.round()
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class MultiSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Attention_block(nn.Module):
    def __init__(self,channels_g,channels_x,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(channels_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(channels_x, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi