import sys
sys.path.insert(0, '../')
import os
import os.path as osp
import shutil
import torch
import numpy as np
from configuration.config import cfg
from operators.mnist_evaluator import MNISTEvaluation
from tensorboardX import SummaryWriter
from utils import update_learning_rate
import helpers.visualize as visualize
import time
from datasets.kth_iterator import get_data_dict, get_next_batch


def irregular_sample_fixed(data, en_timestep, fo_timestep):
    ### data: S x B x C x H x W
    en_data = torch.stack([
        data[int(j - 1)] for j in en_timestep])
    fo_data = torch.stack([
        data[int(j - 1)] for j in fo_timestep])
    return data, en_data, fo_data

def train_valid(iterator, encoder_forecaster, optimizer, criterion):
    train_dataloader = iterator['train_dataloader']
    test_dataloader = iterator['test_dataloader']
    n_train_batches = iterator['n_train_batches']
    n_test_batches = iterator['n_test_batches']
    IN_LEN = cfg.MOVINGMNIST.IN_LEN
    OUT_LEN = cfg.MOVINGMNIST.OUT_LEN
    ### evaluator
    evaluater = MNISTEvaluation(seq_len=OUT_LEN)
    ### Training
    if cfg.MODEL.TRAINING_MODE == "continue":
        path = cfg.MODEL.LOAD_TRAIN_CONTINUE_DIR
        model_params = torch.load(path)
        encoder_forecaster.load_state_dict(model_params['model_state_dict'])
        optimizer.load_state_dict(model_params['optimizer_state_dict'])
        epoch = model_params['epoch']+1
        train_loss = 0.0
        valid_loss = 0.0
        avg_mse = 0.0
        avg_mae = 0.0
        avg_ssim = 0.0
        avg_mse_e = 0.0
        avg_mae_e = 0.0
        avg_ssim_e = 0.0
    else:
        epoch = 0
        train_loss = 0.0
        valid_loss = 0.0
        avg_mse = 0.0
        avg_mae = 0.0
        avg_ssim = 0.0
        avg_mse_e = 0.0
        avg_mae_e = 0.0
        avg_ssim_e = 0.0

    if cfg.MODEL.IRR_MODE == "forecaster":
        fo_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_decoder_100.npz'))['t']
        en_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_encoder_fixed_100.npz'))['t']
    elif cfg.MODEL.IRR_MODE == "encoder":
        fo_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_decoder_fixed_100.npz'))['t']
        en_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_encoder_100.npz'))['t']
    elif cfg.MODEL.IRR_MODE == "both":
        fo_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_decoder_both_100.npz'))['t']
        en_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_encoder_both_100.npz'))['t']
    elif cfg.MODEL.IRR_MODE == "regular":
        fo_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_forecaster_re_100.npz'))['t']
        en_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_encoder_re_100.npz'))['t']
    while epoch < 20:
        print("update learning rate!!!")
        update_learning_rate(optimizer, decay_rate=cfg.MODEL.DECAY_RATE, lowest=cfg.MODEL.LR / 10)
        for timestep_index in range(100):
            ## 1) Load train data sequence
            en_timestep = en_timestep_list[timestep_index]
            fo_timestep = fo_timestep_list[timestep_index]
            for iter_id in range(n_train_batches):
                data_dict = get_data_dict(train_dataloader)
                batch_dict = get_next_batch(data_dict)
                train_sequence = torch.permute(torch.cat(
                    [batch_dict['observed_data'], batch_dict['data_to_predict']], dim=1), (1,0,2,3,4))
                train_batch, input_sequence, target_sequence = \
                    irregular_sample_fixed(train_sequence, en_timestep, fo_timestep)
                train_batch = train_batch.to(cfg.GLOBAL.DEVICE)
                input_sequence = input_sequence.to(cfg.GLOBAL.DEVICE)
                target_sequence = target_sequence.to(cfg.GLOBAL.DEVICE)
                train_en_timestep = torch.from_numpy(en_timestep).to(cfg.GLOBAL.DEVICE)
                train_fo_timestep = torch.from_numpy(fo_timestep).to(cfg.GLOBAL.DEVICE)
                ## 2) Train model
                encoder_forecaster.train()
                ## 3) zero grad optimizer
                optimizer.zero_grad()
                ## 4) Compute the output
                output = encoder_forecaster(input_sequence, train_en_timestep, train_fo_timestep).to(cfg.GLOBAL.DEVICE)
                ## 5) Compute loss function
                loss = criterion(output, target_sequence)
                train_target_numpy = target_sequence.cpu().numpy()
                output_numpy = output.detach().cpu().numpy()
                evaluater.update(train_target_numpy, output_numpy)
                mse, mae, ssim = evaluater.calculate_stat()
                avg_mse += mse
                avg_mae += mae
                avg_ssim += ssim
                ## 6) Back propagation
                loss.backward()
                ## 8) Update optimizer
                optimizer.step()
                train_loss += loss.item()
            print("[{}] Epoch: {}, Time_Iter: {}, Loss: {}".format(time.ctime()[11:19], epoch, timestep_index, loss))
            print("MSE: {}, MAE: {}, SSIM: {}".format(
                avg_mse/n_train_batches, avg_mae/n_train_batches, avg_ssim/n_train_batches))
            avg_mse_e += avg_mse
            avg_mae_e += avg_mae
            avg_ssim_e += avg_ssim
            if (timestep_index + 1) % 25 == 0:
                if cfg.MODEL.IRR_MODE == "forecaster":
                    visualize.save_extrap_images_irr(entire=train_batch, pred=output,
                                                     path=cfg.MODEL.TRAIN_IMAGE_PATH,
                                                     fo_timesteps=fo_timestep,
                                                     total_step=iter_id)
                elif cfg.MODEL.IRR_MODE == "regular":
                    visualize.save_extrap_images_irr_ef(entire=train_batch, input=input_sequence, pred=output,
                                                        path=cfg.MODEL.TRAIN_IMAGE_PATH,
                                                        en_timestep=en_timestep, fo_timesteps=fo_timestep,
                                                        total_step=epoch*25 + timestep_index,
                                                        mode="regular")
                else:
                    visualize.save_extrap_images_irr_ef(entire=train_batch, input=input_sequence, pred=output,
                                                        path=cfg.MODEL.TRAIN_IMAGE_PATH,
                                                        en_timestep=en_timestep, fo_timesteps=fo_timestep,
                                                        total_step=epoch*25 + timestep_index,
                                                        mode=None)
                train_loss /= (25*n_train_batches)
                with open(os.path.join(cfg.MODEL.TRAIN_IMAGE_PATH, 'result.txt'), 'a') as f:
                    f.writelines("Epoch: {}, Iter {}: ; Train_loss: {}\n".format(epoch, epoch*25 + timestep_index, str(train_loss)))
                    f.close()
                train_loss = 0.0
            avg_mse = 0.0
            avg_mae = 0.0
            avg_ssim = 0.0
            timestep_index += 1
        ## 9) Compute Test set
        valid_loss = 0.0
        for validtime_index in range(15):
            en_timestep = en_timestep_list[validtime_index]
            fo_timestep = fo_timestep_list[validtime_index]
            for i in range(n_test_batches):
                with torch.no_grad():
                    test_data_dict = get_data_dict(test_dataloader)
                    test_batch_dict = get_next_batch(test_data_dict)
                    test_sequence = torch.permute(torch.cat(
                        [test_batch_dict['observed_data'], test_batch_dict['data_to_predict']], dim=1), (1, 0, 2, 3, 4))
                    test_batch, input_sequence, target_sequence = \
                        irregular_sample_fixed(test_sequence, en_timestep, fo_timestep)
                    test_batch = test_batch.to(cfg.GLOBAL.DEVICE)
                    input_sequence = input_sequence.to(cfg.GLOBAL.DEVICE)
                    target_sequence = target_sequence.to(cfg.GLOBAL.DEVICE)
                    valid_en_timestep = torch.from_numpy(en_timestep_list[validtime_index]).to(cfg.GLOBAL.DEVICE)
                    valid_fo_timestep = torch.from_numpy(fo_timestep_list[validtime_index]).to(cfg.GLOBAL.DEVICE)
                    encoder_forecaster.eval()
                    valid_output = encoder_forecaster(input_sequence, valid_en_timestep, valid_fo_timestep).to(cfg.GLOBAL.DEVICE)
                    loss_v = criterion(valid_output, target_sequence)
                    valid_loss += loss_v.item()
            validtime_index += 1
        valid_loss /= (15*n_test_batches)
        with open(os.path.join(cfg.MODEL.TRAIN_IMAGE_PATH, 'result.txt'), 'a') as f:
            f.writelines("Validation task: Epoch: {}; Valid_loss: {}\n".format(epoch,str(valid_loss)))
            f.close()

        torch.save({
            'epoch': epoch,
            'model_state_dict': encoder_forecaster.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'valid_loss':valid_loss,
            'metrics':[avg_mse/cfg.MOVINGMNIST.TRAIN.SAVE_ITER, avg_mae/cfg.MOVINGMNIST.TRAIN.SAVE_ITER,
                       avg_ssim/cfg.MOVINGMNIST.TRAIN.SAVE_ITER],
            'timestep': fo_timestep
        },
                   osp.join(cfg.MODEL.SAVE_CHECKPOINT_DIR, 'encoder_forecaster_{}.pth'.format(epoch)))
        with open(os.path.join(cfg.MODEL.TRAIN_IMAGE_PATH, 'result.txt'), 'a') as f:
            f.writelines("Evaluation metrics: AVG_MSE: {}; AVG_MAE: {}; AVG_SSIM: {}\n".format(
                avg_mse_e / (100*n_train_batches), avg_mae_e /(100*n_train_batches),
                avg_ssim_e / (100*n_train_batches)))
            f.close()
        avg_mse_e = 0.0
        avg_mae_e = 0.0
        avg_ssim_e = 0.0
        epoch += 1
