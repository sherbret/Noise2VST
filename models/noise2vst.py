#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def augmentation(x, k=0, inverse=False):
    k = k % 8
    if inverse: k = [0, 1, 6, 3, 4, 5, 2, 7][k]
    if k % 2 == 1: x = torch.flip(x, dims=[3])
    return torch.rot90(x, k=(k//2) % 4, dims=[2,3])

def randint(n):
    return int(torch.randint(n, (1,)))

class Donut3x3(nn.Module):
    """ Donut kernel as recommended by Noise2Self """
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, 3, padding=1, padding_mode="reflect", bias=False, groups=channel)
        self.conv.weight.data = torch.zeros_like(self.conv.weight.data)
        self.conv.weight.data[:, :, 0, 1] = 0.25
        self.conv.weight.data[:, :, 1, 0] = 0.25
        self.conv.weight.data[:, :, 1, 2] = 0.25
        self.conv.weight.data[:, :, 2, 1] = 0.25
        self.conv.requires_grad_(False)
    
    def forward(self, x):
        return self.conv(x)

class Spline(nn.Module):
    """ Spline with linear extrapolation """
    def __init__(self, nb_knots=20, x_min=None, x_max=None, is_strictly_increasing=True):
        super().__init__()
        self.nb_knots = nb_knots
        self.x_min = 0 if x_min is None else x_min
        self.x_max = 1 if x_max is None else x_max
        assert self.x_min < self.x_max

        self.alpha = nn.Parameter(torch.zeros(1)) # for inverse transform 
        self.beta = nn.Parameter(torch.zeros(1)) # for inverse transform 

        self.is_strictly_increasing = is_strictly_increasing
        self.eps = 1e-6 # guarantees strict monotonicity

        # Initialization of parameters theta
        if self.is_strictly_increasing:
            self.theta = nn.Parameter(self.y2theta(torch.linspace(self.x_min, self.x_max, self.nb_knots))) 
        else: 
            self.theta = nn.Parameter(torch.linspace(self.x_min, self.x_max, self.nb_knots)) 

    def theta2y(self, theta):
        if not self.is_strictly_increasing:
            return theta
        theta0, theta1 = torch.split(theta, [1, self.nb_knots-1], dim=0)
        return torch.cumsum(torch.cat((theta0, theta1.exp() + self.eps), dim=0), dim=0)

    def y2theta(self, y):
        if not self.is_strictly_increasing:
            return y
        return torch.cat((y[:1], torch.log(y[1:] - y[:-1] - self.eps)), dim=0)

    def forward(self, z, inverse=False):
        if inverse:
            assert self.is_strictly_increasing
            bias = self.alpha * z + self.beta

        z_input_size = z.size()
        z = z.flatten()
        y = self.theta2y(self.theta)

        if not inverse:
            z_norm = (self.nb_knots - 1) * (z - self.x_min) / (self.x_max - self.x_min)
            with torch.no_grad():
                i = torch.floor(z_norm).clip(min=0, max=self.nb_knots-2).long()
            y_left = torch.gather(y, dim=0, index=i)
            y_right = torch.gather(y, dim=0, index=i+1)
            t = z_norm - i
            z = y_left * (1-t) + y_right * t
        else:
            with torch.no_grad():
                i = torch.searchsorted(y, z).clip(min=1, max=self.nb_knots-1).long()
            y_left = torch.gather(y, dim=0, index=i-1)
            y_right = torch.gather(y, dim=0, index=i)
            t =  (z - y_left) / (y_right - y_left)
            z = (self.x_max - self.x_min) / (self.nb_knots-1) * (i - 1 + t) + self.x_min
        
        z = z.view(z_input_size)
        return z if not inverse else z + bias

    # def forward(self, z, inverse=False):
    #     if inverse:
    #         assert self.is_strictly_increasing
    #         bias = self.alpha * z + self.beta

    #     z_input_size = z.size()
    #     z = z.flatten()

    #     x = torch.linspace(self.x_min, self.x_max, self.nb_knots, device=z.device)
    #     y = self.theta2y(self.theta)

    #     with torch.no_grad():
    #         i = torch.searchsorted(x, z) if not inverse else torch.searchsorted(y, z)
    #         i = i.clip(min=1, max=self.nb_knots-1).long()
        
    #     y1 = torch.gather(y, dim=0, index=i-1)
    #     y2 = torch.gather(y, dim=0, index=i)
    #     x1 = torch.gather(x, dim=0, index=i-1)
    #     x2 = torch.gather(x, dim=0, index=i)
    #     z = (y2 - y1) / (x2 - x1) * (z - x1) + y1 if not inverse else (x2 - x1) / (y2 - y1) * (z - y1) + x1 
    #     z = z.view(z_input_size)
    #     return z if not inverse else z + bias

class Noise2VST(nn.Module):
    def __init__(self, nb_knots=128, inverse=True):
        super().__init__()
        self.inverse = inverse
        self.spline1 = Spline(nb_knots, 0, 1)
        if not self.inverse:
            self.spline2 = Spline(nb_knots, 0, 1)
  
    def fit(self, z, denoiser, nb_iterations=2000, patch_size=64, batch_size=4, stride=4, lr=1e-2):
        # Normalization between 0 and 1
        z_min, z_max = z.min(), z.max()
        lam = (z_max - z_min) if z_max > z_min else 1
        z = (z - z_min) / lam

        N, C, H, W = z.shape
    
        parameters = [{"params":  self.spline1.parameters()}]
        if not self.inverse:
            parameters += [{"params":  self.spline2.parameters()}]
        optimizer = optim.Adam(parameters, lr=lr)
        scheduler = StepLR(optimizer, nb_iterations//3, gamma=0.1)

        conv = Donut3x3(C).to(z.device)
        
        for k in range(1, nb_iterations+1):
            optimizer.zero_grad()
            batch = torch.empty(batch_size, C, patch_size, patch_size, device=z.device)
            for b in range(batch_size):
                n, i, j = randint(N), randint(H - patch_size + 1), randint(W - patch_size + 1)
                patch = augmentation(z[n:n+1, :, i:i+patch_size, j:j+patch_size], randint(8))
                batch[b:b+1, ...] = patch

            mask = torch.zeros_like(batch)
            s = stride
            i, j, c = randint(s), randint(s), randint(C)
            mask[:, c:c+1, i::s, j::s] = 1.0
            vst_batch = self.spline1(batch)

            if self.inverse:
                den = self.spline1(denoiser(conv(vst_batch) * mask + vst_batch * (1-mask)), inverse=True)
            else:
                den = self.spline2(denoiser(conv(vst_batch) * mask + vst_batch * (1-mask)))

            loss = F.mse_loss(den[:, c:c+1, i::s, j::s], batch[:, c:c+1, i::s, j::s])

            loss.backward()
            optimizer.step()
            scheduler.step()

    def forward(self, z, denoiser):
        z_min, z_max = z.min(), z.max()
        lam = (z_max - z_min) if z_max > z_min else 1
        z = (z - z_min) / lam
        
        if self.inverse:
            output = self.spline1(denoiser(self.spline1(z)), inverse=True)
        else:
            output = self.spline2(denoiser(self.spline1(z)))
        return output * lam + z_min

        
