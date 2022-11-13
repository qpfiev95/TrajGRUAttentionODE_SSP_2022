import matplotlib

matplotlib.use('Agg')

import torch
from torchvision.utils import save_image
import os
import numpy as np

import utils as utils


def save_test_images(dataset, preds, batch_dict, path, index, input_norm=False):
    preds = preds.cpu().detach()
    if dataset == 'hurricane':
        gt = batch_dict['orignal_data_to_predict'].cpu().detach()
    else:
        gt = batch_dict['data_to_predict'].cpu().detach()

    b, t, c, h, w = gt.shape

    if input_norm:
        preds = utils.denorm(preds)
        gt = utils.denorm(gt)

    os.makedirs(os.path.join(path, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(path, 'gt'), exist_ok=True)

    for i in range(b):
        for j in range(t):
            save_image(preds[i, j, ...], os.path.join(path, 'pred', f"pred_{index + i:03d}_{j:03d}.png"))
            save_image(gt[i, j, ...], os.path.join(path, 'gt', f"gt_{index + i:03d}_{j:03d}.png"))


def make_save_sequence(dataset, batch_dict, res, irregular=False, extrap=True, input_norm=False):
    """ 4 cases: (interp, extrap) | (regular, irregular) """

    b, t, c, h, w = batch_dict['observed_data'].size()

    # Filter out / Select by mask
    if irregular:
        observed_mask = batch_dict["observed_mask"]
        mask_predicted_data = batch_dict["mask_predicted_data"]
        selected_timesteps = int(observed_mask[0].sum())

        if dataset in ['hurricane']:
            batch_dict['observed_data'] = \
                batch_dict['observed_data'][observed_mask.squeeze(-1).byte(), ...].view(b,selected_timesteps, c, h,w)
            batch_dict['data_to_predict'] = batch_dict['data_to_predict'][
                mask_predicted_data.squeeze(-1).byte(), ...].view(b, selected_timesteps, c, h, w)
        else:
            batch_dict['observed_data'] = batch_dict['observed_data'] * observed_mask.unsqueeze(-1).unsqueeze(-1)
            batch_dict['data_to_predict'] = batch_dict['data_to_predict'] * mask_predicted_data.unsqueeze(-1).unsqueeze(
                -1)

    # Make sequence to save
    pred = res['pred_y'].cpu().detach()

    if extrap:
        inputs = batch_dict['observed_data'].cpu().detach()
        gt_to_predict = batch_dict['data_to_predict'].cpu().detach()
        gt = torch.cat([inputs, gt_to_predict], dim=1)
    else:
        gt = batch_dict['data_to_predict'].cpu().detach()

    time_steps = None

    if input_norm:
        gt = utils.denorm(gt)
        pred = utils.denorm(pred)

    return gt, pred, time_steps

def save_extrap_images(gt, pred, path, total_step, input_norm=False):
    pred = torch.permute(pred, (1, 0, 2, 3, 4))
    gt = torch.permute(gt, (1, 0, 2, 3, 4))
    pred = pred.cpu().detach()
    gt = gt.cpu().detach()
    b, t, c, h, w = gt.shape

    # Padding zeros
    PAD = torch.zeros((b, gt.size(1) - pred.size(1), c, h, w))
    pred = torch.cat([PAD, pred], dim=1)

    save_me = []
    for i in range(min([b, 4])):  # save only 4 items
        row = torch.cat([gt[i], pred[i]], dim=0)
        if input_norm:
            row = utils.denorm(row)
        if row.size(1) == 1:
            row = row.repeat(1, 3, 1, 1)
        save_me += [row]
    save_me = torch.cat(save_me, dim=0)
    save_image(save_me, os.path.join(path, f"image_{(total_step + 1):08d}.png"), nrow=t)

def save_extrap_images_irr(entire, pred, path, fo_timesteps, total_step):
    entire = torch.permute(entire, (1, 0, 2, 3, 4))
    entire = entire.cpu().detach()
    pred = torch.permute(pred, (1, 0, 2, 3, 4))
    pred = pred.cpu().detach()
    b, t, c, h, w = entire.shape

    # Padding zeros
    pred_new = torch.zeros((b, t, c, h, w))
    #pred = torch.cat([PAD, pred], dim=1)
    for i,j in enumerate(fo_timesteps):
        pred_new[:, int(j-1),...] = pred[:,i,...]
    save_me = []
    for i in range(min([b, 4])):  # save only 4 items
        row = torch.cat([entire[i], pred_new[i]], dim=0)
        if row.size(1) == 1:
            row = row.repeat(1, 3, 1, 1)
        save_me += [row]
    save_me = torch.cat(save_me, dim=0)
    save_image(save_me, os.path.join(path, f"image_{(total_step + 1):08d}.png"), nrow=t)

def save_extrap_images_irr_ef(entire, input, pred, path,
                              en_timestep, fo_timesteps, total_step, mode=None):
    entire = torch.permute(entire, (1, 0, 2, 3, 4))
    entire = entire.cpu().detach()
    b, t, c, h, w = entire.shape
    pred = torch.permute(pred, (1, 0, 2, 3, 4))
    pred = pred.cpu().detach()
    input = torch.permute(input, (1, 0, 2, 3, 4))
    input = input.cpu().detach()
    L = input.size(1) + pred.size(1)
    if mode == "regular":
        entire = entire[:,:L,...]
        b, t, c, h, w = entire.shape
    # Padding zeros
    total_timestep = np.concatenate([en_timestep, fo_timesteps])
    input_pred = torch.cat([input, pred], dim=1)
    output = torch.zeros((b, t, c, h, w))
    #pred = torch.cat([PAD, pred], dim=1)
    for i,j in enumerate(total_timestep):
        output[:, int(j-1),...] = input_pred[:,i,...]
    save_me = []
    for i in range(min([b, 4])):  # save only 4 items
        row = torch.cat([entire[i], output[i]], dim=0)
        if row.size(1) == 1:
            row = row.repeat(1, 3, 1, 1)
        save_me += [row]
    save_me = torch.cat(save_me, dim=0)
    save_image(save_me, os.path.join(path, f"image_{(total_step + 1):08d}.png"), nrow=t)

def save_interp_images(gt, pred, path, total_step, input_norm=False):
    pred = pred.cpu().detach()
    data = gt.cpu().detach()
    b, t, c, h, w = data.shape

    save_me = []
    for i in range(min([b, 4])):  # save only 4 items
        row = torch.cat([data[i], pred[i]], dim=0)
        if input_norm:
            row = utils.denorm(row)
        if row.size(1) == 1:
            row = row.repeat(1, 3, 1, 1)
        save_me += [row]
    save_me = torch.cat(save_me, dim=0)
    save_image(save_me, os.path.join(path, f"image_{(total_step + 1):08d}.png"), nrow=t)

def save_test_images(frame, dframe, path, input_norm=False):
    frame = torch.permute(frame, (1, 0, 2, 3, 4))
    dframe = torch.permute(dframe, (1, 0, 2, 3, 4))
    frame = frame.cpu().detach()
    dframe = dframe.cpu().detach()
    b, t, c, h, w = frame.shape

    save_me = []
    for i in range(min([b, 4])):  # save only 4 items
        row = torch.cat([frame[i], dframe[i]], dim=0)
        if input_norm:
            row = utils.denorm(row)
        if row.size(1) == 1:
            row = row.repeat(1, 3, 1, 1)
        save_me += [row]
    save_me = torch.cat(save_me, dim=0)
    save_image(save_me, os.path.join(path, f"image_test.png"), nrow=t)


if __name__ == '__main__':
    pass