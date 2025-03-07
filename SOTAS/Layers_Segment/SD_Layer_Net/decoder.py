import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *
from .film import *

class Decoder(nn.Module):
    def __init__(self, n_anatomical_factors : int, n_latent : int):
        super(Decoder, self).__init__()
        n_filters = 16
        self.film_fusion = MultiSequential(
            film_layer(n_anatomical_factors, n_latent=n_latent, n_filters=n_filters),
            film_layer(n_filters, n_latent=n_latent, n_filters=n_filters),
            film_layer(n_filters, n_latent=n_latent, n_filters=n_filters),
            film_layer(n_filters, n_latent=n_latent, n_filters=n_filters)
        )
        self.reconstruction = nn.Sequential(
            nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, anatomical_factors : torch.Tensor, modalities : torch.Tensor):
        fusioned, _ = self.film_fusion(anatomical_factors, modalities)
        reconstructed = self.reconstruction(fusioned)
        return reconstructed
    