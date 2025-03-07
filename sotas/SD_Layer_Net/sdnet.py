import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *
from .modality_encoder import ModalityEncoder
from .decoder import Decoder

from .unet import AttU_Net
from .layer_engine import LayerEngine

class SDNet(nn.Module):
    def __init__(self, img_size=256, n_encoder_latent=15, n_classes=4, n_anatomical_factors=12, layer_engine=1, drop_rate=0.0):
    # def __init__(self, img_size : int, n_encoder_latent : int, n_classes : int, n_anatomical_factors : int, use_segmentor : bool, layer_engine : int, drop_rate : float):
        super(SDNet, self).__init__()
        
        # self.use_segmentor = use_segmentor
        self.n_classes = n_classes
        self.extra_factors = n_anatomical_factors - n_classes
        
        self.u_net = nn.Sequential(
            AttU_Net(1, output_ch=64, act = nn.ReLU, drop_rate=drop_rate, channels=[32, 64, 128, 256, 512])
        )
        
        self.layer_predictor = nn.Sequential(
                conv_block(64, 32, act = nn.ReLU, drop_rate=0.0, kernel_size=11),
                nn.Conv2d(32, n_classes - 1, kernel_size=1,stride=1,padding=0)
            )
        if self.extra_factors > 0:
            self.surface_predictor = nn.Sequential(
                    conv_block(64, 32, act = nn.ReLU, drop_rate=0.0, kernel_size=11),
                    nn.Conv2d(32, n_anatomical_factors - n_classes, kernel_size=1,stride=1,padding=0)
                )
        
        self.softmax = nn.Softmax(1)
        self.modality_encoder = ModalityEncoder(n_anatomical_factors=n_anatomical_factors, img_size=img_size, n_latent=n_encoder_latent)
        self.decoder = Decoder(n_anatomical_factors=n_anatomical_factors, n_latent=n_encoder_latent)
        self.sigmoid = nn.Sigmoid()
        self.layer_engine = LayerEngine(img_size, n_classes)
        
        
    
    def get_modalities(self, input_img, anatomy_factors):
        z_mean, z_logvar, sampled_z = self.modality_encoder(input_img, anatomy_factors)
        return z_mean, z_logvar, sampled_z
    
    def get_reconstructed_img(self, hard_anatomy : torch.Tensor, modalities : torch.Tensor):
        return self.decoder(hard_anatomy, modalities)
    
    def get_z_estimate(self, reconstructed_img, anatomy_factors):
        z_mean, _, _ = self.modality_encoder(reconstructed_img, anatomy_factors)
        return z_mean
    
    def get_layer_anatomical_factors(self, input_img):
        features = self.u_net(input_img)
        layers = self.layer_predictor(features)
        prob_map, layer_positions, clean_masks, extra_losses = self.layer_engine(layers)
        if self.extra_factors > 0:
            surfaces = self.surface_predictor(features)
            non_layers = self.sigmoid(surfaces)
            anatomy_factors = torch.cat([clean_masks, non_layers], dim = 1)
        else:
            anatomy_factors = clean_masks
        hard_anatomy = DifferentiableRounding().apply(anatomy_factors)
        
        return prob_map, layer_positions, clean_masks, hard_anatomy, extra_losses
    
    def get_anatomical_factors(self, input_img):
        soft_anatomy = self.u_net(input_img)
        layers = self.softmax(soft_anatomy[:, :(self.n_classes), ...])
        non_layers = self.softmax(soft_anatomy[:, (self.n_classes):, ...])
        soft_anatomy = torch.cat([layers, non_layers], dim = 1)
        hard_anatomy = DifferentiableRounding().apply(soft_anatomy)
        return soft_anatomy, hard_anatomy
    
    def get_masks(self, soft_anatomy, hard_anatomy):
        pred_mask = soft_anatomy[:, :(self.n_classes), ...]
        return pred_mask
            
        
    
    
    