from configuration.config import cfg
import torch
import numpy as np
import helpers.visualize as visualize
import time
import os
from datasets.mnist_iterator import MovingMNISTAdvancedIterator
from datasets.kth_iterator import parse_datasets
from datasets.kth_iterator import get_data_dict, get_next_batch

def irregular_sample_fixed(data, en_timestep, fo_timestep):
    ### data: S x B x C x H x W
    en_data = torch.stack([
        data[int(j - 1)] for j in en_timestep])
    fo_data = torch.stack([
        data[int(j - 1)] for j in fo_timestep])
    return data, en_data, fo_data

def test_mnist(model, irr_mode, evaluator, criterion):
    ### Iterator
    if cfg.MODEL.DATASET == "MovingMNIST++":
        mnist_iter = MovingMNISTAdvancedIterator(
            distractor_num=cfg.MOVINGMNIST.DISTRACTOR_NUM,
            initial_velocity_range=(cfg.MOVINGMNIST.VELOCITY_LOWER,
                                    cfg.MOVINGMNIST.VELOCITY_UPPER),
            rotation_angle_range=(cfg.MOVINGMNIST.ROTATION_LOWER,
                                  cfg.MOVINGMNIST.ROTATION_UPPER),
            scale_variation_range=(cfg.MOVINGMNIST.SCALE_VARIATION_LOWER,
                                   cfg.MOVINGMNIST.SCALE_VARIATION_UPPER),
            illumination_factor_range=(cfg.MOVINGMNIST.ILLUMINATION_LOWER,
                                       cfg.MOVINGMNIST.ILLUMINATION_UPPER))
        mnist_iter.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, "mnistplus_test_30_5000.npz"))
    else:
        mnist_iter = MovingMNISTAdvancedIterator(
            digit_num=2,
            distractor_num=0,
            initial_velocity_range=(0.0, 3.6),
            rotation_angle_range=(0.0, 0.0),
            scale_variation_range=(1.0, 1.0),
            illumination_factor_range=(1.0, 1.0))
        mnist_iter.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, "mnist_test_30_5000.npz"))
    ### Loading trained params
    path = cfg.MODEL.LOAD_TEST_DIR
    model_params = torch.load(path)
    model.load_state_dict(model_params['model_state_dict'])
    model_params = torch.load(path)
    test_loss = 0.0
    avg_mae = 0.0
    avg_mse = 0.0
    avg_ssim = 0.0
    avg_psnr = 0.0
    if irr_mode == "regular":
        fo_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_decoder_re_100.npz'))['t']
        en_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_encoder_re_100.npz'))['t']
    elif irr_mode == "irregular":
        fo_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_decoder_both_100.npz'))['t']
        en_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_encoder_both_100.npz'))['t']
    timestep_index = 0
    for iter_id in range(cfg.MODEL.TEST.TEST_NUM):
        if timestep_index == 100:
            timestep_index = 0
        en_timestep = en_timestep_list[timestep_index]
        fo_timestep = fo_timestep_list[timestep_index]
        test_frame_dat, test_en_dat, test_fo_dat = mnist_iter.irregular_sample_fixed(
            batch_size=cfg.MOVINGMNIST.TRAIN.BATCH_SIZE,
            en_timestep=en_timestep,
            fo_timestep=fo_timestep,
            seqlen=30,
            random=False)
        test_batch = torch.from_numpy(test_frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
        test_batch /= test_batch.clone().max()
        test_data = torch.from_numpy(test_en_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
        test_data /= test_data.clone().max()
        test_target = torch.from_numpy(test_fo_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
        test_target /= test_target.clone().max()
        test_en_timestep = torch.from_numpy(en_timestep).to(cfg.GLOBAL.DEVICE)
        test_fo_timestep = torch.from_numpy(fo_timestep).to(cfg.GLOBAL.DEVICE)
        timestep_index += 1
        test_target_numpy = test_target.cpu().numpy()
        ###
        with torch.no_grad():
            model.eval()
            test_output = model(test_data, test_en_timestep, test_fo_timestep)
            test_output_numpy = test_output.cpu().numpy()
            ## Evaluation
            evaluator.update(test_target_numpy, test_output_numpy)
            mse, mae, ssim, psnr = evaluator.calculate_stat_test()
            avg_mse += mse
            avg_mae += mae
            avg_ssim += ssim
            avg_psnr += psnr
            # evaluater.clear_all()
            test_loss = criterion(test_output, test_target)
            if (iter_id + 1) % 2 == 0:
                print("[{}] Iter: {}, Loss: {}".format(time.ctime()[11:19], iter_id, test_loss.item()))
                print("MSE: {}, MAE: {}, SSIM: {}, PSNR: {}".format(mse, mae, ssim, psnr))
        if (iter_id + 1) % 5 == 0:
            visualize.save_extrap_images_irr_ef(entire=test_batch, input=test_data, pred=test_output,
                                                path=cfg.MODEL.TEST_IMAGE_PATH,
                                                en_timestep=en_timestep, fo_timesteps=fo_timestep,
                                                total_step=iter_id,
                                                mode=None)
    avg_mse /= cfg.MODEL.TEST.TEST_NUM
    avg_mae /= cfg.MODEL.TEST.TEST_NUM
    avg_ssim /= cfg.MODEL.TEST.TEST_NUM
    avg_psnr /= cfg.MODEL.TEST.TEST_NUM
    print("avg_MSE: {}, avg_MAE: {}, avg_SSIM: {}".format(avg_mse, avg_mae, avg_ssim))
    with open(os.path.join(cfg.MODEL.TEST_IMAGE_PATH, 'result.txt'), 'a') as f:
        f.writelines("avg_MSE: {}, avg_MAE: {}, avg_SSIM: {}, avg_PSNR: {}".format(avg_mse, avg_mae, avg_ssim, avg_psnr))
        f.close()

def test_kth(model, irr_mode, evaluator, criterion):
    ### Fixed test data
    kth_iter = parse_datasets(device=cfg.GLOBAL.DEVICE, batch_size=4, sample_size=30)
    test_set = kth_iter['test_dataloader']
    n_test_batches = kth_iter['n_test_batches']
    ### Loading trained params
    path = cfg.MODEL.LOAD_TEST_DIR
    model_params = torch.load(path)
    model.load_state_dict(model_params['model_state_dict'])
    model_params = torch.load(path)
    test_loss = 0.0
    avg_mae = 0.0
    avg_mse = 0.0
    avg_ssim = 0.0
    avg_psnr = 0.0
    if irr_mode == "regular":
        fo_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_decoder_re_100.npz'))['t']
        en_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_encoder_re_100.npz'))['t']
    elif irr_mode == "irregular":
        fo_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_decoder_both_100.npz'))['t']
        en_timestep_list = np.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, 'timesteps_encoder_both_100.npz'))['t']
    for timestep_index in range(20):
        en_timestep = en_timestep_list[timestep_index]
        fo_timestep = fo_timestep_list[timestep_index]
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
                valid_en_timestep = torch.from_numpy(en_timestep_list[timestep_index]).to(cfg.GLOBAL.DEVICE)
                valid_fo_timestep = torch.from_numpy(fo_timestep_list[timestep_index]).to(cfg.GLOBAL.DEVICE)
                model.eval()
                test_output = model(input_sequence, valid_en_timestep, valid_fo_timestep).to(
                    cfg.GLOBAL.DEVICE)
                loss_v = criterion(test_output, target_sequence)
                test_loss += loss_v.item()
                test_target_numpy = target_sequence.cpu().numpy()
                test_output_numpy = test_output.cpu().numpy()
                evaluator.update(test_target_numpy, test_output_numpy)
                mse, mae, ssim, psnr = evaluator.calculate_stat_test()
            avg_mse += mse
            avg_mae += mae
            avg_ssim += ssim
            avg_ssim += ssim
            avg_psnr += psnr
            # evaluater.clear_all()
            print("[{}] Timestep: {}, iter: {}, Loss: {}".format(time.ctime()[11:19], timestep_index, i, test_loss))
            print("MSE: {}, MAE: {}, SSIM: {}, PSNR: {}".format(mse, mae, ssim, psnr))
            visualize.save_extrap_images_irr_ef(entire=test_batch, input=input_sequence, pred=test_output,
                                                path=cfg.MODEL.TEST_IMAGE_PATH,
                                                en_timestep=en_timestep, fo_timesteps=fo_timestep,
                                                total_step=i + timestep_index * n_test_batches,
                                                mode=None)
    avg_mse /= 20 * n_test_batches
    avg_mae /= 20 * n_test_batches
    avg_ssim /= 20 * n_test_batches
    avg_psnr /= 20 * n_test_batches
    print("avg_MSE: {}, avg_MAE: {}, avg_SSIM: {}".format(avg_mse, avg_mae, avg_ssim))
    with open(os.path.join(cfg.MODEL.TEST_IMAGE_PATH, 'result.txt'), 'a') as f:
        f.writelines("avg_MSE: {}, avg_MAE: {}, avg_SSIM: {}, avg_PSNR: {}".format(avg_mse, avg_mae, avg_ssim, avg_psnr))
        f.close()