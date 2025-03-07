import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
items = current_dir.split('/')
for item in items:
    if 'v2024' in item or  'V2024' in item:
        obj_name = item
obj_name = os.path.abspath('./DWT_2023/{}'.format(obj_name)) # add current object name to sys
sys.path.append(obj_name)

# from models.dwt_models.DWT_IDWT_layer import *
from models.blocks.dwt_modules.DWT_IDWT_layer import *


"""
@article{Wang_2022_WAT,  
title={Wavelet attention network for the segmentation of layer structures on OCT images}, 
url={http://dx.doi.org/10.1364/boe.475272}, 
DOI={10.1364/boe.475272}, 
journal={Biomedical Optics Express}, 
author={Wang, Cong and Gan, Meng},
year={2022}, 
month={Dec}, 
pages={6167},  
language={en-US}  }
"""
def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return down_conv

class WAT(nn.Module):
    def __init__(self,num_channels, reduction_ratio=2):
        super(WAT,self).__init__()
    
        self.DWT_down = DWT_2D() 
        
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        
        cA,cH,cV,cD = self.DWT_down(x)
        
        F_dwt = torch.add(cA,cH)
        batch_size, num_channels, H, W = F_dwt.size()
        squeeze_tensor = F_dwt.view(batch_size, num_channels, -1).mean(dim=2)
        
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(x, fc_out_2.view(a, b, 1, 1))
        return output_tensor 
    
class WATNet(nn.Module):
    def __init__(self,in_channels = 3, num_classes =4):
        super(WATNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)          
        self.start_conv = x2conv(in_channels, 64)  
        
        self.DWT1 = WAT(64)
        self.conv1 = x2conv(64, 128) 
        
        self.DWT2 = WAT(128)
        self.conv2 = x2conv(128, 256) 
        
        self.DWT3 = WAT(256)
        self.conv3 = x2conv(256, 512) 
        
        self.DWT4 = WAT(512)
        self.conv4 = x2conv(512, 1024) 
        
        self.middle_conv = x2conv(1024, 1024)  
        
        self.uppool4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)       
        self.dec_conv4 = x2conv(1024, 512)       
        
        self.uppool3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)        
        self.dec_conv3 = x2conv(512, 256) 
        
        self.uppool2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)        
        self.dec_conv2 = x2conv(256, 128) 
        
        self.uppool1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)        
        self.dec_conv1 = x2conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    
    def forward(self,x):
        
        x1 = self.start_conv(x)  #B,64,512,512
        x1 = self.DWT1(x1)        
       
        x2 = self.Maxpool(x1) #B, 64, 256,256         
        x2 = self.conv1(x2) #B, 128, 256,256
        x2 = self.DWT2(x2)
        
        
        x3 = self.Maxpool(x2)          
        x3 = self.conv2(x3) #B, 512, 128,128
        x3 = self.DWT3(x3)
        
        
        x4 = self.Maxpool(x3)       
        x4 = self.conv3(x4)  #B,1024,64,64
        x4 = self.DWT4(x4)
        
        x5 = self.Maxpool(x4) 
        x5 = self.conv4(x5)  #B,1024,32,32                
        x5 = self.middle_conv(x5) #B,1024,32,32
        
        d4 = self.uppool4(x5) #B, 512, 64, 64        
        d4 = torch.cat((x4,d4), dim=1) # B, 1024, 128,128
        d4 = self.dec_conv4(d4)
        d4 = self.DWT4(d4)
        
        d3 = self.uppool3(d4)        
        d3 = torch.cat((x3,d3), dim=1)
        d3 = self.dec_conv3(d3) 
        d3 = self.DWT3(d3)
        
        d2 = self.uppool2(d3)        
        d2 = torch.cat((x2,d2),dim=1)
        d2 = self.dec_conv2(d2)
        d2 = self.DWT2(d2)
        
        d1 = self.uppool1(d2)
        d1 = torch.cat((x1,d1),dim=1)
        d1 = self.dec_conv1(d1)
        d1 = self.DWT1(d1)
               
        d = self.final_conv(d1)  
        return d

if __name__=='__main__': 
    import os
    import sys
    CUDA_LAUNCH_BLOCKING=1
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    
    

    # from models.sota_models.MWG_Net.VGGNET import VGG

    model =WATNet(in_channels=3, num_classes=4).cuda()
    
    input = torch.randn(1,3,512,512)
    
    input = input.cuda()
    
    out = model(input)
    
    print(out)