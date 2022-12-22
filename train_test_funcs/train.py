from utils import update_learning_rate
import helpers.visualize as visualize
from configuration.config import cfg
import torch
import numpy as np
import os
import time
import os.path as osp
from datasets.kth_iterator import get_data_dict, get_next_batch

print('test 2')
def train_mnist(model, train_set, valid_set, irr_mode, criterion, optimizer, evaluator):
    ### Training
    print('test 3')
    if cfg.MODEL.TRAINING_MODE == "continue":
        path = cfg.MODEL.LOAD_TRAIN_CONTINUE_DIR
        model_params = torch.load(path)
        model.load_state_dict(model_params['model_state_dict'])
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
    # Time sampling
    if irr_mode == "regular":
        fo_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_decoder_re_100.npz'))['t']
        en_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_encoder_re_100.npz'))['t']
    elif irr_mode == "irregular":
        fo_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_decoder_both_100.npz'))['t']
        en_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_encoder_both_100.npz'))['t']
    timestep_index = 0
    while iter_id < cfg.MODEL.TRAIN.MAX_ITER:
        if (iter_id + 1) % (cfg.MODEL.TRAIN.UPDATE_ITER_INTERVAL) == 0:
            print("update learning rate!!!")
            update_learning_rate(optimizer, decay_rate=cfg.MODEL.DECAY_RATE, lowest=cfg.MODEL.LR/10)
        if timestep_index == 100:
            timestep_index = 0
        ## 1) Load train data sequence
        en_timestep = en_timestep_list[timestep_index]
        fo_timestep = fo_timestep_list[timestep_index]
        frame_dat, en_dat, fo_dat = train_set.irregular_sample_fixed(
            batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
            en_timestep=en_timestep,
            fo_timestep=fo_timestep,
            seqlen=30,
            random=True)
        train_batch = torch.from_numpy(frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
        train_batch /= train_batch.clone().max()
        train_data = torch.from_numpy(en_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
        train_data /= train_data.clone().max()
        train_target = torch.from_numpy(fo_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
        train_target /= train_target.clone().max()
        train_en_timestep = torch.from_numpy(en_timestep).to(cfg.GLOBAL.DEVICE)
        train_fo_timestep = torch.from_numpy(fo_timestep).to(cfg.GLOBAL.DEVICE)
        timestep_index += 1
        ## 2) Train model
        model.train()
        ## 3) zero grad optimizer
        optimizer.zero_grad()
        ## 4) Compute the output
        output = model(train_data, train_en_timestep, train_fo_timestep)
        ## 5) Compute loss function
        loss = criterion(output, train_target)
        train_target_numpy = train_target.cpu().numpy()
        output_numpy = output.detach().cpu().numpy()
        ## 6) Update evaluation
        evaluator.update(train_target_numpy, output_numpy)
        mse, mae, ssim = evaluator.calculate_stat()
        avg_mse += mse
        avg_mae += mae
        avg_ssim += ssim
        if (iter_id + 1) % cfg.MODEL.TRAIN.PRINT_ITER_INTERVAL == 0:
            print("[{}] Iter: {}, Loss: {}".format(time.ctime()[11:19], iter_id, loss))
            print("MSE: {}, MAE: {}, SSIM: {}".format(mse, mae, ssim))
        ## 7) Back propagation
        loss.backward()
        ## 8) Update optimizer
        optimizer.step()
        train_loss += loss.item()
        if (iter_id+1) % (cfg.MODEL.TRAIN.TEST_ITER_INTERVAL ) == 0:  ### 2000
            #train_mse, train_mae, train_ssim = evaluater.calculate_stat()
            train_loss = train_loss/cfg.MODEL.TRAIN.TEST_ITER_INTERVAL
            evaluator.clear_all()
            if cfg.MODEL.IRR_MODE == "forecaster":
                visualize.save_extrap_images_irr(entire=train_batch, pred=output,
                                             path=cfg.MODEL.TRAIN_IMAGE_PATH,
                                             fo_timesteps=fo_timestep,
                                             total_step=iter_id)
            elif cfg.MODEL.IRR_MODE == "regular":
                visualize.save_extrap_images_irr_ef(entire=train_batch, input=train_data, pred=output,
                                                 path=cfg.MODEL.TRAIN_IMAGE_PATH,
                                                 en_timestep=en_timestep, fo_timesteps=fo_timestep,
                                                 total_step=iter_id,
                                                 mode = "regular")
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
            ## 9) Compute validation
            if (iter_id + 1) % (cfg.MODEL.TRAIN.VALID_ITER_INTERVAL) == 0:  ### 8000 The duration of one epoch
                valid_loss = 0.0
                if cfg.MODEL.DATASET == "MovingMNIST++":
                    valid_set.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, "mnistplus_validation_30_2000.npz"))
                else:
                    valid_set.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, "mnist_validation_30_2000.npz"))
                validtime_index = 0
                for i in range(cfg.MOVINGMNIST.VALID.VALID_NUM):
                    if validtime_index == 100:
                        validtime_index = 0
                    with torch.no_grad():
                        valid_frame_dat, valid_en_dat, valid_fo_dat = valid_set.irregular_sample_fixed(
                            batch_size=cfg.MOVINGMNIST.TRAIN.BATCH_SIZE,
                            en_timestep=en_timestep_list[validtime_index],
                            fo_timestep=fo_timestep_list[validtime_index],
                            seqlen=cfg.MODEL.TOTAL_LEN,
                            random=False)
                        valid_batch = torch.from_numpy(valid_frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
                        valid_batch /= valid_batch.max()
                        valid_data = torch.from_numpy(valid_en_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
                        valid_data /= valid_data.max()
                        valid_target = torch.from_numpy(valid_fo_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
                        valid_target /= valid_target.max()
                        valid_en_timestep = torch.from_numpy(en_timestep_list[validtime_index]).to(cfg.GLOBAL.DEVICE)
                        valid_fo_timestep = torch.from_numpy(fo_timestep_list[validtime_index]).to(cfg.GLOBAL.DEVICE)
                        model.eval()
                        valid_output = model(valid_data, valid_en_timestep, valid_fo_timestep)
                        ####
                        loss_v = criterion(valid_output, valid_target)
                        valid_loss += loss_v.item()
                    validtime_index += 1
                valid_loss /= cfg.MODEL.VALID.VALID_NUM
                with open(os.path.join(cfg.MODEL.TRAIN_IMAGE_PATH, 'result.txt'), 'a') as f:
                    f.writelines("Validation task: Iter {}: ; Valid_loss: {}\n".format(iter_id, str(valid_loss)))
                    f.close()
            ## 10) Save learnable params
            if (iter_id + 1) % (cfg.MODEL.TRAIN.SAVE_ITER) == 0:  ### 8000
                # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
                torch.save({
                    'iter': iter_id + 2,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'valid_loss': valid_loss,
                    'metrics': [avg_mse / cfg.MODEL.TRAIN.SAVE_ITER, avg_mae / cfg.MODEL.TRAIN.SAVE_ITER,
                                avg_ssim / cfg.MODEL.TRAIN.SAVE_ITER],
                    'en_timestep': en_timestep,
                    'fo_timestep': fo_timestep

                },
                    osp.join(cfg.MODEL.SAVE_CHECKPOINT_DIR, 'encoder_forecaster_{}.pth'.format(iter_id)))
                with open(os.path.join(cfg.MODEL.TRAIN_IMAGE_PATH, 'result.txt'), 'a') as f:
                    f.writelines("Evaluation metrics: AVG_MSE: {}; AVG_MAE: {}; AVG_SSIM: {}\n".format(
                        avg_mse / cfg.MODEL.TRAIN.SAVE_ITER, avg_mae / cfg.MODEL.TRAIN.SAVE_ITER,
                        avg_ssim / cfg.MODEL.TRAIN.SAVE_ITER))
                    f.close()
                avg_mse = 0.0
                avg_mae = 0.0
                avg_ssim = 0.0
        iter_id += 1

def irregular_sample_fixed(data, en_timestep, fo_timestep):
    ### data: S x B x C x H x W
    en_data = torch.stack([
        data[int(j - 1)] for j in en_timestep])
    fo_data = torch.stack([
        data[int(j - 1)] for j in fo_timestep])
    return data, en_data, fo_data

def train_kth(model, dataset, irr_mode, criterion, optimizer, evaluator):
    train_set = dataset['train_dataloader']
    test_set = dataset['test_dataloader']
    n_train_batches = dataset['n_train_batches']
    n_test_batches = dataset['n_test_batches']
    ### Training
    if cfg.MODEL.TRAINING_MODE == "continue":
        path = cfg.MODEL.LOAD_TRAIN_CONTINUE_DIR
        model_params = torch.load(path)
        model.load_state_dict(model_params['model_state_dict'])
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
    # Time sampling
    if irr_mode == "regular":
        fo_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_decoder_re_100.npz'))['t']
        en_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_encoder_re_100.npz'))['t']
    elif irr_mode == "irregular":
        fo_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_decoder_both_100.npz'))['t']
        en_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_encoder_both_100.npz'))['t']
    while epoch < cfg.MODEL.TRAIN.NUM_EPOCHS:
        print("update learning rate!!!")
        update_learning_rate(optimizer, decay_rate=cfg.MODEL.DECAY_RATE, lowest=cfg.MODEL.LR / 10)
        for timestep_index in range(100):
            ## 1) Load train data sequence
            en_timestep = en_timestep_list[timestep_index]
            fo_timestep = fo_timestep_list[timestep_index]
            for iter_id in range(n_train_batches):
                data_dict = get_data_dict(train_set)
                batch_dict = get_next_batch(data_dict)
                train_sequence = torch.permute(torch.cat(
                    [batch_dict['observed_data'], batch_dict['data_to_predict']], dim=1), (1, 0, 2, 3, 4))
                train_batch, input_sequence, target_sequence = \
                    irregular_sample_fixed(train_sequence, en_timestep, fo_timestep)
                train_batch = train_batch.to(cfg.GLOBAL.DEVICE)
                input_sequence = input_sequence.to(cfg.GLOBAL.DEVICE)
                target_sequence = target_sequence.to(cfg.GLOBAL.DEVICE)
                train_en_timestep = torch.from_numpy(en_timestep).to(cfg.GLOBAL.DEVICE)
                train_fo_timestep = torch.from_numpy(fo_timestep).to(cfg.GLOBAL.DEVICE)
                ## 2) Train model
                model.train()
                ## 3) zero grad optimizer
                optimizer.zero_grad()
                ## 4) Compute the output
                output = model(input_sequence, train_en_timestep, train_fo_timestep).to(cfg.GLOBAL.DEVICE)
                ## 5) Compute loss function
                loss = criterion(output, target_sequence)
                train_target_numpy = target_sequence.cpu().numpy()
                output_numpy = output.detach().cpu().numpy()
                evaluator.update(train_target_numpy, output_numpy)
                mse, mae, ssim = evaluator.calculate_stat()
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
                avg_mse / n_train_batches, avg_mae / n_train_batches, avg_ssim / n_train_batches))
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
                                                        total_step=epoch * 25 + timestep_index,
                                                        mode="regular")
                else:
                    visualize.save_extrap_images_irr_ef(entire=train_batch, input=input_sequence, pred=output,
                                                        path=cfg.MODEL.TRAIN_IMAGE_PATH,
                                                        en_timestep=en_timestep, fo_timesteps=fo_timestep,
                                                        total_step=epoch * 25 + timestep_index,
                                                        mode=None)
                train_loss /= (25 * n_train_batches)
                with open(os.path.join(cfg.MODEL.TRAIN_IMAGE_PATH, 'result.txt'), 'a') as f:
                    f.writelines("Epoch: {}, Iter {}: ; Train_loss: {}\n".format(epoch, epoch * 25 + timestep_index, str(train_loss)))
                    f.close()
                train_loss = 0.0
            avg_mse = 0.0
            avg_mae = 0.0
            avg_ssim = 0.0
            timestep_index += 1

            ## 9) Compute Test set
        valid_loss = 0.0
        for validtime_index in range(10):
            en_timestep = en_timestep_list[validtime_index]
            fo_timestep = fo_timestep_list[validtime_index]
            for i in range(n_test_batches):
                with torch.no_grad():
                    test_data_dict = get_data_dict(test_set)
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
                    model.eval()
                    valid_output = model(input_sequence, valid_en_timestep, valid_fo_timestep).to(cfg.GLOBAL.DEVICE)
                    loss_v = criterion(valid_output, target_sequence)
                    valid_loss += loss_v.item()
            validtime_index += 1
        valid_loss /= (10 * n_test_batches)
        with open(os.path.join(cfg.MODEL.TRAIN_IMAGE_PATH, 'result.txt'), 'a') as f:
            f.writelines("Validation task: Epoch {}: ; Valid_loss: {}\n".format(epoch, str(valid_loss)))
            f.close()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'valid_loss': valid_loss,
            'metrics': [avg_mse / cfg.MODEL.TRAIN.SAVE_ITER, avg_mae / cfg.MODEL.TRAIN.SAVE_ITER,
                        avg_ssim / cfg.MODEL.TRAIN.SAVE_ITER],
            'timestep': fo_timestep
        },
            osp.join(cfg.MODEL.SAVE_CHECKPOINT_DIR, 'encoder_forecaster_{}.pth'.format(epoch)))
        with open(os.path.join(cfg.MODEL.TRAIN_IMAGE_PATH, 'result.txt'), 'a') as f:
            f.writelines("Evaluation metrics: AVG_MSE: {}; AVG_MAE: {}; AVG_SSIM: {}\n".format(
                avg_mse_e / (100 * n_train_batches), avg_mae_e / (100 * n_train_batches),
                avg_ssim_e / (100 * n_train_batches)))
            f.close()
        avg_mse_e = 0.0
        avg_mae_e = 0.0
        avg_ssim_e = 0.0
        epoch += 1


