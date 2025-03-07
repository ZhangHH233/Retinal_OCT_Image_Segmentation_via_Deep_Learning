import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *
import numpy as np

class LayerEngine(nn.Module):
    def __init__(self, img_size : int, n_classes : int):
        super(LayerEngine, self).__init__()
        self.n_classes = n_classes
        self.img_size = img_size
        self.softmax = nn.Softmax(2)
        self.logsoftmax = nn.LogSoftmax(2)
        self.relu = nn.ReLU()
        self.pos_mask = {}
        
        self.sobel_kernel = torch.tensor([[[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]], dtype=torch.float32, requires_grad=False).cuda()
        self.sobel_kernel = torch.repeat_interleave(self.sobel_kernel, self.n_classes - 1, dim=0)
        
        self.laplace_kernel = torch.tensor([[[1, 0, 0, 0, 0, -2, 0, 0, 0, 0, 1]]], dtype=torch.float32, requires_grad=False).cuda()
        self.laplace_kernel = torch.repeat_interleave(self.laplace_kernel, self.n_classes - 1, dim=0)
        self.curv_max = self.get_max_layer_curv_max()
        
    def get_position_mask(self, template):
        w = template.shape[-1]
        if w in self.pos_mask:
            return self.pos_mask[w]
        column = np.arange(template.shape[-2])
        column = np.expand_dims(column, 1)
        mask = np.repeat(column, template.shape[-1], axis=1)
        mask = np.expand_dims(mask, 0)
        mask = np.repeat(mask, self.n_classes - 1, axis=0)
        mask = np.expand_dims(mask, axis=0)
        self.pos_mask[w] = torch.tensor(mask, dtype=torch.float32, requires_grad=False).cuda()
        return self.pos_mask[w]
    
    def get_max_layer_curv_max(self):

        # Change this to the number of your layers accordingly
        curv_max = torch.tensor([[1.2261], [1.1558], [1.1161], [1.1195], [2.7202], [2.3714], [1.7055], [3.2717], [2.6716], [5.0418], [0.4293]], dtype=torch.float32, requires_grad=False)

        curv_max = torch.repeat_interleave(curv_max, self.img_size, dim=1)
        return curv_max.cuda()
    

    def get_layer_positions(self, sm):
        return torch.sum(sm * self.get_position_mask(sm), dim=2)
    
    def get_cumulative_mask(self, sm):
        batch_size, _, w, h = sm.shape
        upper_mask = torch.ones((batch_size, 1, w, h), dtype=sm.dtype, device=sm.device)
        cum_mask = torch.cumsum(sm, dim=2)
        return torch.cat([upper_mask, cum_mask], dim=1)
        
    
    def topological_engine_2d(self, cum_mask):
        for i in range(self.n_classes - 2):
            idx = i + 2
            cum_mask[:, idx, ...] = self.relu(cum_mask[:, idx, ...] + cum_mask[:, idx - 1, ...] - 1)
        return cum_mask
    
    def topological_engine_1d(self, layer_positions):
        new_positions = layer_positions.clone()
        for i in range(self.n_classes - 2):
            idx = i + 1
            new_positions[:, idx, ...] = new_positions[:, idx - 1, ...] + self.relu(layer_positions[:, idx, ...] - new_positions[:, idx - 1, ...])
        return new_positions
    
    def separate_masks(self, cum_mask):
        for idx in range(self.n_classes - 1):
            cum_mask[:, idx, ...] = cum_mask[:, idx, ...] - cum_mask[:, idx + 1, ...]
        return cum_mask
    
    def get_topology_violations(self, layer_positions):
        violations = layer_positions[:, :-1, ...] - layer_positions[:, 1:, ...]
        return self.relu(violations)
    
    def get_standard_deviations(self, sm, layer_positions):
        return torch.sqrt(torch.sum(sm * ((self.get_position_mask(sm) - layer_positions.unsqueeze(2))**2), dim=2))
    
    def get_curvature_diffs(self, layer_positions):
        pad_length = self.sobel_kernel.shape[2] // 2
        pad = (pad_length, pad_length)
        layer_positions_pad = F.pad(layer_positions, pad, 'replicate')
        first_order_drv = F.conv1d(layer_positions_pad, self.sobel_kernel, groups=self.n_classes-1)
        second_order_drv = F.conv1d(layer_positions_pad, self.laplace_kernel, groups=self.n_classes-1)
        numerator = second_order_drv
        denominator = torch.pow(1 + torch.pow(first_order_drv, 2.0), 1.5)
        curvature = numerator / denominator
        curv_diff = torch.abs(curvature) - self.curv_max
        return curv_diff
    
    def get_neighbour_diff(self, layer_positions):
        return torch.abs((torch.roll(layer_positions, 1, dims=2) - layer_positions)[...,1:])
    
        
    def forward(self, soft_anatomy):
        
        pred_mask = soft_anatomy[:, :(self.n_classes - 1), ...]
        sm = self.softmax(pred_mask)
        lsm = self.logsoftmax(pred_mask)
        layer_positions = self.get_layer_positions(sm)
        std_deviations = self.get_standard_deviations(sm, layer_positions)
        topology_violations = self.get_topology_violations(layer_positions)
        continuity_violations = self.get_neighbour_diff(layer_positions)
        curvature_diffs = self.get_curvature_diffs(layer_positions)
        correct_layer_positions = self.topological_engine_1d(layer_positions)
        
        cum_mask = self.get_cumulative_mask(sm)
        top_correct_mask = self.topological_engine_2d(cum_mask)
        clean_masks = self.separate_masks(top_correct_mask)
        
        losses = {"std_deviations": std_deviations, "topology_violations": topology_violations, "continuity_violations": continuity_violations, "curvature_diffs": curvature_diffs}
        
        return lsm, correct_layer_positions, clean_masks, losses