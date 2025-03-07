import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *

class film_layer(nn.Module):
    def __init__(self, n_input_channels : int, n_latent : int, n_filters : int):
        super(film_layer, self).__init__()
        self.n_filters = n_filters
        
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv2_block = nn.Sequential(
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.pred_dense = nn.Sequential(
            nn.Linear(in_features=n_latent, out_features = 2 * n_filters),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=2 * n_filters, out_features = 2 * n_filters),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
        )
        self.final_lr = nn.LeakyReLU(negative_slope=0.2)
        
        
    def film_connect(self, spatial_data, gamma, beta):
        spatial_shape = spatial_data.shape
        gamma_shape = gamma.shape
        beta_shape = beta.shape
        
        # tile gamma and beta:
        gamma = torch.tile(torch.reshape(gamma, (gamma_shape[0], gamma_shape[1], 1, 1)),
                        (1, 1, spatial_shape[2], spatial_shape[3]))
        beta = torch.tile(torch.reshape(beta, (beta_shape[0], beta_shape[1], 1, 1)),
                       (1, 1, spatial_shape[2], spatial_shape[3]))

        # compute output:
        return spatial_data * gamma + beta

        
        
    def forward(self, incoming, modalities):
        
        conv1 = self.conv1_block(incoming)
        conv2 = self.conv2_block(conv1)

        dense_out = self.pred_dense(modalities)
        
        gamma_l2, beta_l2 = dense_out[:, :self.n_filters], dense_out[:, self.n_filters:]
        
        film = self.film_connect(conv2, gamma_l2, beta_l2)
        film_act = self.final_lr(film)

        film_sum = conv1 + film_act

        return film_sum, modalities