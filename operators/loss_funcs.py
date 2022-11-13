import numpy as np
from torch import nn
import torch
from configuration.config import cfg
import torch.nn.functional as F
from operators.msssim import _SSIMForMultiScale
from pytorch_msssim import ms_ssim, ssim
from utils import rainfall_to_pixel, dBZ_to_pixel
from math import log10, sqrt

class MyWeightedMseMae(nn.Module):
    def __init__(self, seq_len, w_mae=1, w_mse=1):
        super(MyWeightedMseMae, self).__init__()
        # weight: Tensor: (out_len,1)
        weight = torch.from_numpy(np.arange(1, seq_len + 1).reshape((1, seq_len)))
        self.weight = weight.to(torch.device("cuda"))

    def forward(self, pred, gt):
        pred = torch.mul(pred.permute(1,2,3,4,0), self.weight)
        pred = pred.permute(4,0,1,2,3)
        gt = torch.mul(gt.permute(1,2,3,4,0), self.weight)
        gt = gt.permute(4, 0, 1, 2, 3)
        #mse = torch.sum((pred - gt)**2, (2, 3, 4))
        #mae = torch.sum(torch.abs((pred - gt)), (2, 3, 4))
        mse = torch.square(pred - gt)
        mae = torch.abs(pred - gt)
        return (torch.mean(mse) + torch.mean(mae))

class MyMseMae(nn.Module):
    def __init__(self, seq_len, w_mae=1, w_mse=1):
        super(MyMseMae, self).__init__()


    def forward(self, pred, gt):
        mse = torch.square(pred - gt)
        mae = torch.abs(pred - gt)
        return (torch.mean(mse) + torch.mean(mae))

class MyMseMaeSSIM(nn.Module):
    def __init__(self, seq_len, w_mae=1, w_mse=1):
        super(MyMseMaeSSIM, self).__init__()

    def forward(self, pred, gt):
        #pred = torch.reshape(pred, (-1, pred.size(2), ))
        #print(pred.size(), gt.size())
        mse = torch.square(pred - gt)
        mae = torch.abs(pred - gt)
        SSIM = 1 - ssim(pred, gt, data_range=1)
        #print(torch.mean(mse), torch.mean(mae), SSIM)
        return (torch.mean(mse) + torch.mean(mae) + 0.05*SSIM)

class MyMseMaeSSIMDiff(nn.Module):
    def __init__(self, seq_len, w_mae=1, w_mse=1):
        super(MyMseMaeSSIMDiff, self).__init__()

    def forward(self, pred, gt, diff, gt_diff):
        #pred = torch.reshape(pred, (-1, pred.size(2), ))
        #print(pred.size(), gt.size())
        data_diff = gt_diff[1:,:, ...] - gt_diff[:-1,:, ...]
        mse_diff = torch.square(diff  - data_diff)
        mse = torch.square(pred - gt)
        mae = torch.abs(pred - gt)
        SSIM = 1 - ssim(pred, gt, data_range=1)
        #print(torch.mean(mse), torch.mean(mae), SSIM)
        #return (torch.mean(mse) + torch.mean(mae) + 0.01*SSIM + 0.1*torch.mean(mse_diff))
        return (torch.mean(mse) + torch.mean(mae) + torch.mean(mse_diff))
#def ssim_loss(pred, gt):
#    seq#_len = pred.shape[0]
#    batch_size = pred.shape[1]
#    pred = pred.reshape((pred.shape[0]*pred.shape[1],
#                         pred.shape[3], pred.shape[4],1))
#    gt = gt.reshape((gt.shape[0] * gt.shape[1],
#                         gt.shape[3], gt.shape[4], 1))
#    ssim, cs = _SSIMForMultiScale(img1=pred, img2=gt, max_val=1.0)
#    ret = ssim.reshape((seq_len, batch_size)).sum(axis=1)
#    return ret.mean()

def ssim_loss(pred, gt):
    pred_torch = torch.from_numpy(pred)
    gt_torch = torch.from_numpy(gt)
    ssim_lossf = 1 - ssim(pred_torch, gt_torch, data_range=1) #################
    return np.asarray(ssim_lossf)

def mae_loss(pred, gt):
    l1 = np.abs(pred-gt)
    return l1.mean()

def mse_loss(pred, gt):
    l2 = np.square(pred-gt)
    return l2.mean()
    
def psnr_loss (pred, gt):
    mse = mse_loss(pred, gt)
    if (mse == 0):
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

### HKO loss func
class Weighted_MSE_MAE_hko(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, LAMBDA=None):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self._lambda = LAMBDA

    def forward(self, input, target, mask):
        # __C.HKO.EVALUATION.BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30)
        # __C.HKO.EVALUATION.THRESHOLDS = np.array([0.5, 2, 5, 10, 30]) evaluated in rainfall intensity
        balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
        weights = torch.ones_like(input) * balancing_weights[0] # ones (SxBxCxHxW)
        thresholds = [rainfall_to_pixel(ele) for ele in cfg.HKO.EVALUATION.THRESHOLDS]
        for i, threshold in enumerate(thresholds):
            # assign a weight according to its rainfall intensity
            # (rainfall intensity has already been converted into pixel intensity)
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        weights = weights * mask.float()
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum(weights * ((input-target)**2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input-target))), (2, 3, 4))
        if self._lambda is not None:
            S, B = mse.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(mse.get_device())
            mse = (w * mse.permute(1, 0)).permute(1, 0)
            mae = (w * mae.permute(1, 0)).permute(1, 0)
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))