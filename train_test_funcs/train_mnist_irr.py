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


def train_valid(train_iterator, valid_iterator, encoder_forecaster, optimizer, criterion):
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
        iter_id = model_params['iter']
        train_loss = 0.0
        valid_loss = 0.0
        avg_mse = 0.0
        avg_mae = 0.0
        avg_ssim = 0.0
    else:
        iter_id = 0
        train_loss = 0.0
        valid_loss = 0.0
        avg_mse = 0.0
        avg_mae = 0.0
        avg_ssim = 0.0
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
    timestep_index = 0
    while iter_id < cfg.MOVINGMNIST.TRAIN.MAX_ITER:
        if (iter_id + 1) % (cfg.MODEL.UPDATE_ITER) == 0:
            print("update learning rate!!!")
            update_learning_rate(optimizer, decay_rate=cfg.MODEL.DECAY_RATE, lowest=cfg.MODEL.LR/10)
        if timestep_index == 100:
            timestep_index = 0
        ## 1) Load train data sequence
        en_timestep = en_timestep_list[timestep_index]
        fo_timestep = fo_timestep_list[timestep_index]
        #frame_dat, fo_dat = train_iterator.irregular_forecaster_sample_fixed(
        #    batch_size=cfg.MOVINGMNIST.TRAIN.BATCH_SIZE,
        #    fo_timestep=timestep,
        #    seqlen=30,
        #    random=True)
        frame_dat, en_dat, fo_dat = train_iterator.irregular_sample_fixed(
            batch_size=cfg.MOVINGMNIST.TRAIN.BATCH_SIZE,
            en_timestep=en_timestep,
            fo_timestep=fo_timestep,
            seqlen=30,
            random=True)

        train_batch = torch.from_numpy(frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
        train_batch /= train_batch.clone().max()
        #train_data = train_batch[:IN_LEN, ...]
        train_data = torch.from_numpy(en_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
        train_data /= train_data.clone().max()
        train_target = torch.from_numpy(fo_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
        train_target /= train_target.clone().max()
        train_en_timestep = torch.from_numpy(en_timestep).to(cfg.GLOBAL.DEVICE)
        train_fo_timestep = torch.from_numpy(fo_timestep).to(cfg.GLOBAL.DEVICE)
        timestep_index += 1
        ## 2) Train model
        encoder_forecaster.train()
        ## 3) zero grad optimizer
        optimizer.zero_grad()
        ## 4) Compute the output
        #t_en = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float)
        #encoder_timestep = torch.from_numpy(t_en).to(cfg.GLOBAL.DEVICE)
        output = encoder_forecaster(train_data, train_en_timestep, train_fo_timestep)
        ## 5) Compute loss function
        loss = criterion(output, train_target)
        train_target_numpy = train_target.cpu().numpy()
        output_numpy = output.detach().cpu().numpy()
        ## 6) Update evaluation
        evaluater.update(train_target_numpy, output_numpy)
        mse, mae, ssim = evaluater.calculate_stat()
        avg_mse += mse
        avg_mae += mae
        avg_ssim += ssim
        if (iter_id + 1) % cfg.MOVINGMNIST.TRAIN.PRINT_ITER_INTERVAL == 0:
            print("[{}] Iter: {}, Loss: {}".format(time.ctime()[11:19], iter_id, loss))
            print("MSE: {}, MAE: {}, SSIM: {}".format(mse, mae, ssim))
        ## 7) Back propagation
        loss.backward()
        ## 8) Update optimizer
        optimizer.step()
        train_loss += loss.item()
        if (iter_id+1) % (cfg.MOVINGMNIST.TRAIN.TEST_ITER_INTERVAL) == 0:  ### 2000
            #train_mse, train_mae, train_ssim = evaluater.calculate_stat()
            train_loss = train_loss/cfg.MOVINGMNIST.TRAIN.TEST_ITER_INTERVAL
            evaluater.clear_all()
            if cfg.MODEL.IRR_MODE == "forecaster":
                visualize.save_extrap_images_irr(entire=train_batch, pred=output,
                                             path=cfg.MODEL.TRAIN_IMAGE_PATH,
                                             fo_timesteps=fo_timestep,
                                             total_step=iter_id,
                                             mode = "forecaster")
            elif cfg.MODEL.IRR_MODE == "regular":
                visualize.save_extrap_images_irr(entire=train_batch, pred=output,
                                                 path=cfg.MODEL.TRAIN_IMAGE_PATH,
                                                 fo_timesteps=fo_timestep,
                                                 total_step=iter_id,
                                                 mode="regular")
            else:
                visualize.save_extrap_images_irr_ef(entire=train_batch, input=train_data, pred=output,
                                                 path=cfg.MODEL.TRAIN_IMAGE_PATH,
                                                 en_timestep=en_timestep, fo_timesteps=fo_timestep,
                                                 total_step=iter_id,
                                                 mode = None)

            with open(os.path.join(cfg.MODEL.TRAIN_IMAGE_PATH, 'result.txt'), 'a') as f:
                f.writelines("Iter {}: ; Train_loss: {}\n".format(iter_id,str(train_loss)))
                f.close()
            train_loss = 0.0

        if (iter_id + 1) % (cfg.MOVINGMNIST.TRAIN.UPDATE_ITER_INTERVAL) == 0:  ### 8000 The duration of one epoch
            ## 9) Compute validation
            valid_loss = 0.0
            if cfg.MOVINGMNIST.PLUS == True:
                valid_iterator.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, "mnistplus_validation_30_2000.npz"))
            else:
                valid_iterator.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, "mnist_validation_30_2000.npz"))
            validtime_index = 0
            for i in range(cfg.MOVINGMNIST.VALID.VALID_NUM):
                if validtime_index == 100:
                    validtime_index = 0
                with torch.no_grad():
                    #valid_frame_dat, valid_fo_dat = valid_iterator.irregular_forecaster_sample_fixed(
                    #    batch_size=cfg.MOVINGMNIST.TRAIN.BATCH_SIZE,
                    #    fo_timestep=timestep_list[validtime_index],
                    #    seqlen=30,
                    #    random=False)
                    valid_frame_dat, valid_en_dat, valid_fo_dat = valid_iterator.irregular_sample_fixed(
                        batch_size=cfg.MOVINGMNIST.TRAIN.BATCH_SIZE,
                        en_timestep=en_timestep_list[validtime_index],
                        fo_timestep=fo_timestep_list[validtime_index],
                        seqlen=30,
                        random=False)
                    valid_batch = torch.from_numpy(valid_frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
                    valid_batch /= valid_batch.max()
                    #valid_data = valid_batch[:IN_LEN, ...]
                    valid_data = torch.from_numpy(valid_en_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
                    valid_data /= valid_data.max()
                    valid_target = torch.from_numpy(valid_fo_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
                    valid_target /= valid_target.max()
                    valid_en_timestep = torch.from_numpy(en_timestep_list[validtime_index]).to(cfg.GLOBAL.DEVICE)
                    valid_fo_timestep = torch.from_numpy(fo_timestep_list[validtime_index]).to(cfg.GLOBAL.DEVICE)
                    encoder_forecaster.eval()
                    valid_output = encoder_forecaster(valid_data, valid_en_timestep, valid_fo_timestep)
                    loss_v = criterion(valid_output, valid_target)
                    valid_loss += loss_v.item()
                validtime_index += 1
            valid_loss /= cfg.MOVINGMNIST.VALID.VALID_NUM
            with open(os.path.join(cfg.MODEL.TRAIN_IMAGE_PATH, 'result.txt'), 'a') as f:
                f.writelines("Validation task: Iter {}: ; Valid_loss: {}\n".format(iter_id,str(valid_loss)))
                f.close()

        if (iter_id+1) % (cfg.MOVINGMNIST.TRAIN.SAVE_ITER) == 0:  ### 8000
            # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
            torch.save({
                'iter': iter_id+2,
                'model_state_dict': encoder_forecaster.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'valid_loss':valid_loss,
                'metrics':[avg_mse/cfg.MOVINGMNIST.TRAIN.SAVE_ITER, avg_mae/cfg.MOVINGMNIST.TRAIN.SAVE_ITER,
                           avg_ssim/cfg.MOVINGMNIST.TRAIN.SAVE_ITER],
                'timestep': fo_timestep
            },
                       osp.join(cfg.MODEL.SAVE_CHECKPOINT_DIR, 'encoder_forecaster_{}.pth'.format(iter_id)))
            with open(os.path.join(cfg.MODEL.TRAIN_IMAGE_PATH, 'result.txt'), 'a') as f:
                f.writelines("Evaluation metrics: AVG_MSE: {}; AVG_MAE: {}; AVG_SSIM: {}\n".format(
                    avg_mse / cfg.MOVINGMNIST.TRAIN.SAVE_ITER, avg_mae / cfg.MOVINGMNIST.TRAIN.SAVE_ITER,
                    avg_ssim / cfg.MOVINGMNIST.TRAIN.SAVE_ITER))
                f.close()
            avg_mse = 0.0
            avg_mae = 0.0
            avg_ssim = 0.0

        iter_id += 1