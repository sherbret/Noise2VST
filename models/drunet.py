#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# Code inspired from https://github.com/cszn/DPIR/blob/master/models/network_unet.py

class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.res = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False, padding_mode='zeros'),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False, padding_mode='zeros'))
        
    def forward(self, x):
        return x + self.res(x)

class DRUNet(nn.Module):
    def __init__(self, color=False):
        super().__init__()
        in_nc = 3 if color else 1
        out_nc = in_nc
        nc=[64, 128, 256, 512]
        nb=4

        self.m_head = nn.Conv2d(in_nc+1, nc[0], 3, stride=1, padding=1, bias=False)
        
        self.m_down1 = nn.Sequential(
            *[ResBlock(nc[0], nc[0]) for _ in range(nb)],
            nn.Conv2d(nc[0], nc[1], kernel_size=2, stride=2, padding=0, bias=False),
            )
        
        self.m_down2 = nn.Sequential(
            *[ResBlock(nc[1], nc[1]) for _ in range(nb)],
            nn.Conv2d(nc[1], nc[2], kernel_size=2, stride=2, padding=0, bias=False),
            )
        
        self.m_down3 = nn.Sequential(
            *[ResBlock(nc[2], nc[2]) for _ in range(nb)],
            nn.Conv2d(nc[2], nc[3], kernel_size=2, stride=2, padding=0, bias=False),
            )
        
        self.m_body = nn.Sequential(*[ResBlock(nc[3], nc[3]) for _ in range(nb)])
        
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(nc[3], nc[2], kernel_size=2, stride=2, padding=0, bias=False),
            *[ResBlock(nc[2], nc[2]) for _ in range(nb)] 
            )
        
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(nc[2], nc[1], kernel_size=2, stride=2, padding=0, bias=False),
            *[ResBlock(nc[1], nc[1]) for _ in range(nb)] 
            )
        
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(nc[1], nc[0], kernel_size=2, stride=2, padding=0, bias=False),
            *[ResBlock(nc[0], nc[0]) for _ in range(nb)] 
            )
        
        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, stride=1, padding=1, bias=False)
        
    def forward(self, x, sigma=25/255, norm_equiv=True):
        if norm_equiv:
            x_min, x_max = x.min(), x.max()
            lam = (x_max - x_min) if x_max > x_min else 1
            x = (x - x_min) / lam
            sigma = sigma / lam

        # Size handling (h and w must divisible by 8)
        _, _, h, w = x.size()
        r1, r2 = h % 8, w % 8
        x = F.pad(x, pad=(0, 8-r2 if r2 > 0 else 0, 0, 8-r1 if r1 > 0 else 0), mode='constant', value=float(x.mean()))  
        
        noise_level_map = sigma * torch.ones_like(x[:, :1, :, :])
        x0 = torch.cat((x, noise_level_map), dim=1)
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)
        return x[..., :h, :w] if not norm_equiv else x[..., :h, :w] * lam + x_min
