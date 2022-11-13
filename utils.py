import numpy as np
from torch import nn
from collections import OrderedDict
from configuration.config import cfg
import cv2
import os.path as osp
import os
import torch
from datasets.mask import read_mask_file

def get_norm_layer(ch):
    norm_layer = nn.BatchNorm2d(ch)
    return norm_layer

def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

def update_learning_rate(optimizer, decay_rate=0.999, lowest=1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr

def count_pixels(name=None):
    png_dir = cfg.HKO_PNG_PATH
    mask_dir = cfg.HKO_MASK_PATH
    counts = np.zeros(256, dtype=np.float128)
    for root, dirs, files in os.walk(png_dir):
        for file_name in files:
            if not file_name.endswith('.png'):
                continue
            tmp_dir = '/'.join(root.split('/')[-3:])
            png_path = osp.join(png_dir, tmp_dir, file_name)
            mask_path = osp.join(mask_dir, tmp_dir, file_name.split('.')[0]+'.mask')
            label, count = np.unique(cv2.cvtColor(cv2.imread(png_path), cv2.COLOR_BGR2GRAY)[read_mask_file(mask_path)], return_counts=True)
            counts[label] += count
    if name is not None:
        np.save(name, counts)
    return counts

def pixel_to_dBZ(img):
    """
    Parameters
    ----------
    img : np.ndarray or float
    Returns
    -------
    """
    return img * 70.0 - 10.0

def dBZ_to_pixel(dBZ_img):
    """
    Parameters
    ----------
    dBZ_img : np.ndarray
    Returns
    -------
    """
    return np.clip((dBZ_img + 10.0) / 70.0, a_min=0.0, a_max=1.0)


def pixel_to_rainfall(img, a=58.53, b=1.56):
    """Convert the pixel values to real rainfall intensity
    Parameters
    ----------
    img : np.ndarray
    a : float32, optional
    b : float32, optional
    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    dBZ = pixel_to_dBZ(img)
    dBR = (dBZ - 10.0 * np.log10(a)) / b
    rainfall_intensity = np.power(10, dBR / 10.0)
    return rainfall_intensity


def rainfall_to_pixel(rainfall_intensity, a=58.53, b=1.56):
    """Convert the rainfall intensity to pixel values
    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional
    Returns
    -------
    pixel_vals : np.ndarray
    """
    dBR = np.log10(rainfall_intensity) * 10.0
    # dBZ = 10b log(R) +10log(a)
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals

def dBZ_to_rainfall(dBZ, a=58.53, b=1.56):
    return np.power(10, (dBZ - 10 * np.log10(a))/(10*b))

def rainfall_to_dBZ(rainfall, a=58.53, b=1.56):
    return 10*np.log10(a) + 10*b*np.log10(rainfall)