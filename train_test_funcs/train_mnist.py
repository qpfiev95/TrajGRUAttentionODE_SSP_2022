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

    ###
    #folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]
    save_dir = cfg.MOVINGMNIST.MODEL_SAVE_DIR
    model_save_dir = osp.join(save_dir, 'models')
    log_dir = osp.join(save_dir, 'logs')
    all_scalars_file_name = osp.join(save_dir, "all_scalars.json")
    pkl_save_dir = osp.join(save_dir, 'pkl')
    if osp.exists(all_scalars_file_name):
        os.remove(all_scalars_file_name)
    if osp.exists(log_dir):
        shutil.rmtree(log_dir)
    if osp.exists(model_save_dir):
        shutil.rmtree(model_save_dir)
    os.mkdir(model_save_dir)
    writer = SummaryWriter(log_dir)

    ###
    #frame_dat, _ = iterator.sample(batch_size=cfg.MOVINGMNIST.TRAIN.BATCH_SIZE,
    #                               seqlen=cfg.MOVINGMNIST.IN_LEN + cfg.MOVINGMNIST.OUT_LEN)
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
    while iter_id < cfg.MOVINGMNIST.TRAIN.MAX_ITER:
        if (iter_id + 1) % (cfg.MODEL.UPDATE_ITER) == 0:
            print("update learning rate!!!")
            update_learning_rate(optimizer, decay_rate=cfg.MODEL.DECAY_RATE, lowest=cfg.MODEL.LR/10)
        ## 1) Load train data sequence
        frame_dat, _ = train_iterator.sample(batch_size=cfg.MOVINGMNIST.TRAIN.BATCH_SIZE,
                                         seqlen=cfg.MOVINGMNIST.IN_LEN + cfg.MOVINGMNIST.OUT_LEN)
        train_batch = torch.from_numpy(frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
        train_batch /= train_batch.clone().max()
        train_data = train_batch[:IN_LEN, ...]
        train_target = train_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
        ## 2) Train model
        encoder_forecaster.train()
        ## 3) zero grad optimizer
        optimizer.zero_grad()
        ## 4) Compute the output
        output = encoder_forecaster(train_data)
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
            train_mse, train_mae, train_ssim = evaluater.calculate_stat()
            train_loss = train_loss/cfg.MOVINGMNIST.TRAIN.TEST_ITER_INTERVAL
            evaluater.clear_all()
            visualize.save_extrap_images(gt=train_batch, pred=output,
                                         path=cfg.MODEL.TRAIN_IMAGE_PATH,
                                         total_step=iter_id)

            with open(os.path.join(cfg.MODEL.TRAIN_IMAGE_PATH, 'result.txt'), 'a') as f:
                f.writelines("Iter {}: ; Train_loss: {}\n".format(iter_id,str(train_loss)))
                f.close()
            train_loss = 0.0

        if (iter_id + 1) % (cfg.MODEL.UPDATE_ITER) == 0:  ### 8000 The duration of one epoch
            ## 9) Compute validation
            valid_loss = 0.0
            for i in range(cfg.MOVINGMNIST.VALID.VALID_NUM):
                with torch.no_grad():
                    ###
                    valid_iterator.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, "mnistplus_validation_20_2000.npz"))
                    ###
                    valid_frame_dat, _ = valid_iterator.sample(batch_size=cfg.MOVINGMNIST.TRAIN.BATCH_SIZE,
                                                         seqlen=cfg.MOVINGMNIST.IN_LEN + cfg.MOVINGMNIST.OUT_LEN,
                                                         random=False)
                    valid_batch = torch.from_numpy(valid_frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
                    valid_batch /= valid_batch.max()
                    valid_data = valid_batch[:IN_LEN, ...]
                    valid_target = valid_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
                    encoder_forecaster.eval()
                    valid_output = encoder_forecaster(valid_data)
                    loss_v = criterion(valid_output, valid_target)
                    valid_loss += loss_v.item()
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
                           avg_ssim/cfg.MOVINGMNIST.TRAIN.SAVE_ITER]
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