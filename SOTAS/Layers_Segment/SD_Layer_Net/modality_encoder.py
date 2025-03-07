import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *

class ModalityEncoder(nn.Module):
    def __init__(self, n_anatomical_factors : int, img_size : int, n_latent : int):
        super(ModalityEncoder, self).__init__()
        n_encoder_channels = 16
        self.input_size = img_size
        self.n_latent = n_latent
        self.encoders = nn.Sequential(
            self._build_encoder(1 + n_anatomical_factors, n_encoder_channels),
            self._build_encoder(n_encoder_channels, n_encoder_channels),
            self._build_encoder(n_encoder_channels, n_encoder_channels),
            self._build_encoder(n_encoder_channels, n_encoder_channels),
        )
        self.dense = nn.Sequential(
            nn.Linear(in_features = n_encoder_channels *(img_size // 16) * (img_size // 16), out_features=32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)            
        )
        self.z_mean = nn.Linear(32, out_features=n_latent)
        self.z_logvar = nn.Linear(32, out_features=n_latent)
        
        
    
    def _build_encoder(self, in_channels : int, out_channels : int):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)            
        )
    
    def _z_distribution(self, encoded : torch.Tensor):
        z_mean = self.z_mean(encoded)
        z_logvar = self.z_logvar(encoded)
        
        eps = torch.randn((encoded.shape[0], self.n_latent), dtype = torch.float32, device=encoded.device)
        
        sampled_z = z_mean + eps * torch.exp(0.5 * z_logvar)
        
        return z_mean, z_logvar, sampled_z
    
    def forward(self, input_data : torch.Tensor, anatomical_factors : torch.Tensor):
        incoming = torch.cat([input_data, anatomical_factors], dim=1)
        encoded = self.encoders(incoming)
        flat_encoded = torch.flatten(encoded, start_dim = 1)
        encoded = self.dense(flat_encoded)
        z_mean, z_logvar, sampled_z = self._z_distribution(encoded)
        return z_mean, z_logvar, sampled_z
    