#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def augmentation(x, k=0, inverse=False):
    """
    Applies a geometric transformation to the input tensor. The function supports 8 distinct 
    transformations defined by `k` in the range [0, 7], combining 4 possible rotations 
    (0°, 90°, 180°, 270°) with or without horizontal flip.

    Parameters:
        x (torch.Tensor): Input tensor of shape (..., H, W), where H and W are spatial dimensions.
        k (int, optional): Transformation index in the range [0, 7]. Defaults to 0.
                           - k // 2 determines the rotation: 0 (0°), 1 (90°), 2 (180°), 3 (270°).
                           - k % 2 determines whether to apply a horizontal flip.
        inverse (bool, optional): If True, applies the inverse of the transformation corresponding 
                                  to `k`. Defaults to False.

    Returns:
        torch.Tensor: Transformed tensor with the same shape as the input.

    Example:
        x_aug = augmentation(x, k=5)        # Apply rotation + flip
        x_orig = augmentation(x_aug, k=5, inverse=True)  # Recover original
    """
    k = k % 8
    if inverse:
        k = [0, 1, 6, 3, 4, 5, 2, 7][k]
    if k % 2 == 1:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k=k//2, dims=[-2, -1])

class Donut3x3(nn.Module):
    """
    Fixed 3x3 convolutional filter with a donut-shaped kernel, as recommended by Noise2Self.

    This module applies a non-trainable depthwise convolution with a predefined "donut"-shaped
    kernel, which averages the four direct neighbors (top, bottom, left, right) of each pixel,
    ignoring the center and diagonal pixels. It is commonly used for smoothing or extracting
    neighbor-based features while preserving the center pixel.

    The kernel has the following fixed structure:
            [[0.0, 0.25, 0.0],
            [0.25, 0.0, 0.25],
            [0.0, 0.25, 0.0]]

    Args:
        channel (int): Number of input (and output) channels. 
                       The filter is applied independently to each channel.

    Attributes:
        conv (nn.Conv2d): Depthwise convolutional layer with fixed weights and `reflect` padding.

    Example:
        donut = Donut3x3(channel=3)
        x = torch.randn(1, 3, 32, 32)  # Example input
        y = donut(x)                  # Apply donut filter
    """
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, 3, padding=1, 
                              padding_mode="reflect", bias=False, groups=channel)
        self.conv.weight.data = torch.zeros_like(self.conv.weight.data)
        self.conv.weight.data[:, :, 0, 1] = 0.25
        self.conv.weight.data[:, :, 1, 0] = 0.25
        self.conv.weight.data[:, :, 1, 2] = 0.25
        self.conv.weight.data[:, :, 2, 1] = 0.25
        self.conv.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)

class Spline(nn.Module):
    """
    Piecewise linear spline transformation with optional monotonicity constraints.

    This module defines a learnable spline function with a fixed number of knots over a bounded 
    input domain. It supports both forward and inverse transformations, with an option to enforce 
    the function to be strictly increasing (monotonic).

    Args:
        nb_knots (int): Number of knots used to define the piecewise linear spline. Defaults to 128.
        x_min (float or None): Minimum value of the input domain. Defaults to 0 if None.
        x_max (float or None): Maximum value of the input domain. Defaults to 1 if None.
        is_strictly_increasing (bool): If True, constrains the spline to be strictly increasing. 
                                       Defaults to True.

    Attributes:
        theta (nn.Parameter): Learnable parameters defining the spline values at knots.
        alpha (nn.Parameter): Linear bias coefficient used in inverse mode to correct 
                              for invertibility shifts.
        beta (nn.Parameter): Linear bias offset used in inverse mode to correct 
                             for invertibility shifts.

    Methods:
        forward(z, inverse=False): Applies the spline transformation (or its inverse 
                                   if `inverse=True`) to the input tensor `z`.

    Example:
        spline = Spline(nb_knots=32, x_min=-3, x_max=3, is_strictly_increasing=True)
        z = torch.randn(100)
        y = spline(z)                   # Apply forward spline transformation
        z_recovered = spline(y, inverse=True)  # Apply inverse transformation

    Notes:
        - When `is_strictly_increasing=True`, the spline is parameterized via a differentiable 
          mapping that ensures strict monotonicity, using exponential reparameterization.
        - Inverse transform requires monotonicity (`is_strictly_increasing=True`).
        - In inverse mode, the output includes a learnable affine bias (`alpha * z + beta`).
    """
    def __init__(self, nb_knots=128, x_min=None, x_max=None, is_strictly_increasing=True):
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
    """
    Noise2VST: Self-supervised denoising using a learnable Variance-Stabilizing Transform (VST).

    This module wraps a denoiser with learnable spline-based forward and (optionally) inverse 
    transformations to stabilize noise variance in the input data. The method is trained 
    without ground truth, using a Noise2Self-like masking strategy and a pair of spline transforms 
    to learn a forward (and optionally inverse) VST.

    Args:
        nb_knots (int): Number of knots for the spline transforms. Defaults to 128.
        inverse (bool): If True, only a single spline (with inverse support) is used.
                        If False, a pair of separate splines (forward and inverse) is used. 
                        Defaults to True.

    Attributes:
        spline1 (Spline): The primary spline used as the forward VST.
        spline2 (Spline): The inverse spline used in non-invertible mode (if `inverse=False`).

    Methods:
        fit(z, denoiser, ...): Trains the spline(s) to adapt to the noise structure of the input `z` 
        using the given denoiser.
        forward(z, denoiser): Applies VST → denoising → inverse VST to an input tensor.

    Args in `fit`:
        z (Tensor): Noisy input image of shape [N, C, H, W].
        denoiser (Callable): A denoising function or network compatible with the shape of `z`.
        nb_iterations (int): Number of optimization iterations. Defaults to 2000.
        patch_size (int): Spatial size of cropped training patches. Defaults to 64.
        batch_size (int): Number of patches per training batch. Defaults to 4.
        stride (int): Pixel stride for the masking pattern. Defaults to 4.
        lr (float): Learning rate for Adam optimizer. Defaults to 1e-2.

    Example:
        noise2VST = Noise2VST(nb_knots=128, inverse=True)
        noise2VST.fit(noisy_image, denoiser=my_denoiser)
        clean_image = noise2VST(noisy_image, denoiser=my_denoiser)

    Notes:
        - If `inverse=True`, `spline1` is used both as the forward and inverse transform (invertible mode).
        - If `inverse=False`, `spline1` and `spline2` are separate and do not require invertibility.
    """
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
                n, i, j = randint(0, N-1), randint(0, H - patch_size), randint(0, W - patch_size)
                patch = augmentation(z[n:n+1, :, i:i+patch_size, j:j+patch_size], randint(0, 7))
                batch[b:b+1, ...] = patch

            mask = torch.zeros_like(batch)
            s = stride
            i, j, c = randint(0, s-1), randint(0, s-1), randint(0, C-1)
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
