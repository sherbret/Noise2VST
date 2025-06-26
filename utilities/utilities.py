#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torchvision.io import read_image
import numpy as np
import matplotlib.pyplot as plt

def load_img(img_path):
    """ load a 8-bit image and convert it to a torch tensor in the [0, 1] interval with NCHW format """
    img = read_image(img_path).float() / 255.0
    return img[None, ...]

def psnr(img_true, img_noisy):
    """ assumes that the images are in [0, 1] """
    mse = F.mse_loss(img_true.clip(0,1), img_noisy.clip(0,1))
    return float(-10*torch.log10(mse))

def show_img(img_list, titles=None, color=False):
    """ assumes that the images are in [0, 1] """
    x_list = []
    with torch.no_grad():
        for img in img_list:
            if not color:
                x_list.append(img[0, 0, ...].cpu().numpy().clip(0, 1)) 
            else:
                x_list.append(img[0, ...].permute(1, 2, 0).cpu().numpy().clip(0, 1))

    fig = plt.figure(figsize=(20, 5))
    rows, columns = 1, len(x_list) # setting values to rows and column variables

    for j in range(columns):
        fig.add_subplot(rows, columns, j+1)
        if not color:
            plt.imshow(x_list[j], cmap='magma')
        else:
            plt.imshow(x_list[j])
        if titles is not None:
            plt.title(titles[j])
    plt.show()
    
def f_GAT(x, a=1.0, b=0.0, target_sigma=1.0):
    if a < 1e-10: return x / np.sqrt(b) * target_sigma
    sigma2 = b / a**2
    z = x / a
    return 2 * torch.sqrt((z + 3/8 + sigma2).clip(min=0)) * target_sigma

def f_GAT_inv(x, a=1.0, b=0.0, target_sigma=1.0):
    x = x / target_sigma
    if a < 1e-10: return x * np.sqrt(b)
    sigma2 = b / a**2
    return a*(0.25 * x**2 + 0.25 * np.sqrt(3/2) * x**-1 - 11/8 * x**-2 + 5/8 * np.sqrt(3/2) * x**-3 - 1/8 - sigma2) # closed-form approximation of the exact unbiased
    # return a * (0.25 * x**2 - 1/8 - sigma2) # good approximation if x is big
    # return a/4 * x**2 - 3/8 * a - b/a # algebraic inverse (poor results)
    
def display_vst(model, ab_GAT=None, target_sigma=25/255):
    """ Display the learned VST of a Noise2VST model """
    # For a normalization-equivariant denoiser D and a constant c, we have g(D(f(x) - c) + c) = g(D(f(x))).
    # It means that replacing f by f - c and g by g(. + c) does not change the final outcome.
    blue, red = (30/255, 144/255, 255/255), (255/255, 16/255, 240/255)
    
    # Display learned VST
    with torch.no_grad():
        x = torch.linspace(0, 1, model.spline1.nb_knots, device=model.spline1.theta.device)
        y = model.spline1(x)
        z = model.spline1(y, inverse=True) if model.inverse else model.spline2(x)
        x, y, z = x.cpu(), y.cpu(), z.cpu()
        c = y.min()
        if model.inverse:
            plt.plot(x, y - c, color=blue, label=r"$f_\theta$")
            plt.plot(y - c, z, color=red, label=r"$f^{inv}_{\theta, \alpha, \beta}$")
        else:
            plt.plot(x, y - c, color=blue, label=r"$f_{\theta_1}$")
            plt.plot(x - c, z, color=red, label=r"$f_{\theta_2}$")
            
    # Display GAT
    if ab_GAT is not None:
        a, b = ab_GAT
        x = torch.linspace(0, 1, 100 * model.spline1.nb_knots, device=model.spline1.theta.device)
        y = f_GAT(x, a, b, target_sigma)
        z = f_GAT_inv(y, a, b, target_sigma)
        x, y, z = x.cpu(), y.cpu(), z.cpu()
        c = y.min()
        plt.plot(x, y - c, "--", color=blue, label=r"$f_{GAT}$")
        plt.plot(y - c, z, "--",  color=red, label=r"$f^{inv}_{GAT}$")
    plt.plot(x, x, "--", color="black", label="identity")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Learned VST by Noise2VST")
    plt.legend()
    plt.show()