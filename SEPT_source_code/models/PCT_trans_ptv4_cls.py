"""PCT_trans_ptv4_cls.py

ModelNet40 语义通信端到端模型。
参考 PCT_trans_ptv3 的轻量化 PU-GAN 解码器，实现编码器-信道-解码器流程。
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from openpoints.models.backbone.pointnet import PointNetEncoder


class AWGNChannel(nn.Module):
    """简单 AWGN 信道模拟。"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, snr: float = None) -> torch.Tensor:
        if snr is None:
            return x
        power = torch.mean(x.pow(2), dim=1, keepdim=True)
        x = x / torch.sqrt(power + 1e-8)
        noise_power = 1.0 / (10 ** (snr / 10.0))
        noise = torch.randn_like(x) * torch.sqrt(torch.tensor(noise_power, device=x.device))
        return x + noise


def calc_cd(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """计算对称 Chamfer Distance."""
    if p1.dim() != 3 or p2.dim() != 3:
        raise ValueError('Expected input tensors shape [B, N, 3]')
    p1_expand = p1.unsqueeze(2)
    p2_expand = p2.unsqueeze(1)
    dist = torch.sum((p1_expand - p2_expand) ** 2, dim=-1)
    dist_p1 = torch.min(dist, dim=2)[0]
    dist_p2 = torch.min(dist, dim=1)[0]
    cd = torch.mean(dist_p1, dim=1) + torch.mean(dist_p2, dim=1)
    return cd


class LightweightPUGANDecoder(nn.Module):
    """轻量化 PU-GAN 风格解码器，将隐向量映射回 2048×3 点云。"""
    def __init__(self, bottleneck_size: int, num_points: int, upsample_factor: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.num_points = num_points
        self.upsample_factor = upsample_factor
        self.coarse_points = max(num_points // upsample_factor, 64)
        self.bottleneck_size = bottleneck_size

        self.coarse_mlp = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.coarse_points * 3)
        )

        self.feat_proj = nn.Conv1d(bottleneck_size, hidden_dim, 1)
        self.upsample_net = nn.Sequential(
            nn.Conv1d(hidden_dim + 3, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3 * upsample_factor, 1)
        )

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        coarse = self.coarse_mlp(x).view(B, self.coarse_points, 3)
        global_feat_proj = self.feat_proj(x.unsqueeze(-1)).expand(-1, -1, self.coarse_points).permute(0, 2, 1)
        combined = torch.cat([global_feat_proj, coarse], dim=2).transpose(1, 2)
        offsets = self.upsample_net(combined)
        offsets = offsets.view(B, 3, self.upsample_factor, self.coarse_points)
        offsets = offsets.permute(0, 3, 2, 1).contiguous()
        coarse_expanded = coarse.unsqueeze(2).repeat(1, 1, self.upsample_factor, 1)
        fine = coarse_expanded + offsets * 0.05
        recon = fine.view(B, -1, 3)
        return recon, coarse


class PointMetaBase_Trans(nn.Module):
    """ModelNet40 端到端语义通信模型。"""
    def __init__(self,
                 in_channels: int = 3,
                 bottleneck_size: int = 300,
                 recon_points: int = 2048,
                 jscc_hidden_dim: int = 512,
                 pointnet_width: int = 64,
                 **kwargs):
        super().__init__()
        self.encoder = PointNetEncoder(
            in_channels=in_channels,
            input_transform=True,
            feature_transform=True,
            is_seg=False
        )
        self.jscc_encoder = nn.Sequential(
            nn.Linear(self.encoder.out_channels, jscc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(jscc_hidden_dim, bottleneck_size)
        )
        self.channel = AWGNChannel()
        self.jscc_decoder = LightweightPUGANDecoder(bottleneck_size, recon_points)

    def forward(self, points, snr=None):
        x_feat = None
        # Accept multiple input formats: tensor, numpy array, or dict containing
        # 'pos' and optional 'x'. Ensure pos and x (if present) are tensors with
        # shape [B, N, 3] / [B, N, C].
        if isinstance(points, dict):
            pos = points.get('pos', None)
            x_feat = points.get('x', None)
        else:
            pos = points

        # Convert numpy or list input to torch if necessary
        if isinstance(pos, (list, tuple, np.ndarray)):
            pos = torch.from_numpy(np.asarray(pos)).float()
        elif not isinstance(pos, torch.Tensor):
            raise TypeError(f'Unsupported pos type: {type(pos)}')
        if isinstance(x_feat, (list, tuple, np.ndarray)):
            x_feat = torch.from_numpy(np.asarray(x_feat)).float()
        elif x_feat is not None and not isinstance(x_feat, torch.Tensor):
            raise TypeError(f'Unsupported x_feat type: {type(x_feat)}')

        if pos is None:
            raise ValueError('输入点云为空 (points 中未找到 pos)。')

        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        # If x_feat missing, let encoder derive it from pos
        if x_feat is not None:
            if x_feat.dim() == 2:
                x_feat = x_feat.unsqueeze(0)
            # Convert [B, N, C] to [B, C, N] for PointNet if necessary.
            if x_feat.dim() == 3 and x_feat.shape[1] == pos.shape[1] and x_feat.shape[2] == pos.shape[2]:
                x_feat = x_feat.permute(0, 2, 1)
            global_feat = self.encoder.forward_cls_feat(pos, x=x_feat)
        else:
            global_feat = self.encoder.forward_cls_feat(pos)
        tx_symbols = self.jscc_encoder(global_feat)
        rx_symbols = self.channel(tx_symbols, snr)
        recon_points, coarse_points = self.jscc_decoder(rx_symbols)
        cd = calc_cd(pos, recon_points)
        coarse_cd = calc_cd(pos, coarse_points)
        return recon_points, cd, coarse_cd


def get_model(bottleneck_size: int = 300,
              recon_points: int = 2048,
              jscc_hidden_dim: int = 512,
              **kwargs):
    return PointMetaBase_Trans(
        bottleneck_size=bottleneck_size,
        recon_points=recon_points,
        jscc_hidden_dim=jscc_hidden_dim,
        **kwargs
    )
