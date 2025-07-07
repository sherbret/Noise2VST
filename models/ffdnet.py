#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# Code inspired from https://github.com/cszn/FFDNet

class FFDNet(nn.Module):
    def __init__(self, color=False):
        super().__init__()
        if color:
            channels_in, channels_out, num_of_layers, features = 13, 12, 12, 96
        else:
            channels_in, channels_out, num_of_layers, features = 5, 4, 15, 64

        layers = []
        layers.append(nn.Conv2d(channels_in, features, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, channels_out, 3, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x, sigma=25/255, norm_equiv=True):
        if norm_equiv:
            x_min, x_max = x.min(), x.max()
            lam = (x_max - x_min) if x_max > x_min else 1
            x = (x - x_min) / lam
            sigma = sigma / lam

        # Size handling (h and w must divisible by 2)
        _, _, h, w = x.size()
        x = F.pad(x, pad=(0, w%2, 0, h%2), mode='constant', value=float(x.mean()))

        noise_level_map = sigma * torch.ones_like(x[:, :1, :, :])
        x = F.pixel_shuffle(self.model(torch.cat((F.pixel_unshuffle(x, 2), F.avg_pool2d(noise_level_map, 2)), dim=1)), 2)
        return x[..., :h, :w] if not norm_equiv else x[..., :h, :w] * lam + x_min

