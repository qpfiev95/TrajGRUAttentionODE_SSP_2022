import sys
import os
sys.path.insert(0, '../../')
import torch
from configuration.config import cfg
from operators.loss_funcs import MyMseMaeSSIM, Weighted_MSE_MAE_hko
from train_test_funcs.train_kth_irr import train_valid
import numpy as np
from train_test_funcs.train import train_mnist, train_kth
from train_test_funcs.test import test_mnist, test_kth

## 0) Params
time_step_encoder = torch.from_numpy(cfg.MODEL.T_EN).to(cfg.GLOBAL.DEVICE)
time_step_decoder = torch.from_numpy(cfg.MODEL.T_DE).to(cfg.GLOBAL.DEVICE)
## 1) Model
if cfg.MODEL.TECH == "ConvGRU":
    from configuration.net_params_ConvGRU import encoder_params, decoder_params
    from model_structure.encoder_decoder_baseline import Encoder, Decoder, ED
    encoder = Encoder(encoder_params[0], encoder_params[1],
                                 time_step_encoder).to(cfg.GLOBAL.DEVICE)
    decoder = Decoder(decoder_params[0], decoder_params[1],
                            time_step_decoder).to(cfg.GLOBAL.DEVICE)
    model = ED(encoder, decoder).to(cfg.GLOBAL.DEVICE)
elif cfg.MODEL.TECH == "TrajGRU":
    from configuration.net_params_TrajGRU import encoder_params, decoder_params
    from model_structure.encoder_decoder_baseline import Encoder, Decoder, ED
    encoder = Encoder(encoder_params[0], encoder_params[1],
                                 time_step_encoder).to(cfg.GLOBAL.DEVICE)
    decoder = Decoder(decoder_params[0], decoder_params[1],
                                 time_step_decoder).to(cfg.GLOBAL.DEVICE)
    model = ED(encoder, decoder).to(cfg.GLOBAL.DEVICE)
elif cfg.MODEL.TECH == "TrajGRU-Attention":
    from configuration.net_params_TrajGRU_Attention import encoder_params, decoder_params
    from model_structure.encoder_decoder_zz_ode import Encoder, Decoder, ED
    encoder = Encoder(encoder_params[0], encoder_params[1], encoder_params[2],
                      time_step_encoder).to(cfg.GLOBAL.DEVICE)
    decoder = Decoder(decoder_params[0], decoder_params[1], decoder_params[2], decoder_params[3],
                            time_step_decoder, False).to(cfg.GLOBAL.DEVICE)
    model = ED(encoder, decoder).to(cfg.GLOBAL.DEVICE)
elif cfg.MODEL.TECH == "TrajGRU-Attention-ODE":
    from configuration.net_params_TrajGRU_Attention_ODE import encoder_params, decoder_params
    from model_structure.encoder_decoder_zz_ode import Encoder, Decoder, ED
    encoder = Encoder(encoder_params[0], encoder_params[1], encoder_params[2],
                      time_step_encoder).to(cfg.GLOBAL.DEVICE)
    decoder = Decoder(decoder_params[0], decoder_params[1], decoder_params[2], decoder_params[3],
                            time_step_decoder, True).to(cfg.GLOBAL.DEVICE)
    model = ED(encoder, decoder).to(cfg.GLOBAL.DEVICE)
elif cfg.MODEL.TECH == "Vid-ODE":
    from model_components.vid_ode import VidODE
    model = VidODE(input_size=cfg.VidODE.INPUT_SIZE,
                   input_dim=cfg.VidODE.INPUT_DIM,
                   init_dim=cfg.VidODE.INIT_DIM,
                   n_downs=cfg.VidODE.NUM_DOWNSAMPLING,
                   n_layers=cfg.VidODE.NUM_LAYERS,
                   device=cfg.GLOBAL.DEVICE)
else:
    print("This model has not implemented yet!!!")
## 2) Dataset
if cfg.MODEL.DATASET == "MovingMNIST":
    from datasets.mnist_iterator import MovingMNISTAdvancedIterator
    train_iter = MovingMNISTAdvancedIterator(
        digit_num=2,
        distractor_num=0,
        initial_velocity_range=(0.0, 3.6),
        rotation_angle_range=(0.0, 0.0),
        scale_variation_range=(1.0, 1.0),
        illumination_factor_range=(1.0, 1.0))
    train_iter.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, "mnist_train_30_10000.npz"))
    valid_iter = MovingMNISTAdvancedIterator(
        digit_num=2,
        distractor_num=0,
        initial_velocity_range=(0.0, 3.6),
        rotation_angle_range=(0.0, 0.0),
        scale_variation_range=(1.0, 1.0),
        illumination_factor_range=(1.0, 1.0))
elif cfg.MODEL.DATASET == "MovingMNIST++":
    from datasets.mnist_iterator import MovingMNISTAdvancedIterator
    train_iter = MovingMNISTAdvancedIterator(
        distractor_num=cfg.MOVINGMNIST.DISTRACTOR_NUM,
        initial_velocity_range=(cfg.MOVINGMNIST.VELOCITY_LOWER,
                                cfg.MOVINGMNIST.VELOCITY_UPPER),
        rotation_angle_range=(cfg.MOVINGMNIST.ROTATION_LOWER,
                              cfg.MOVINGMNIST.ROTATION_UPPER),
        scale_variation_range=(cfg.MOVINGMNIST.SCALE_VARIATION_LOWER,
                               cfg.MOVINGMNIST.SCALE_VARIATION_UPPER),
        illumination_factor_range=(cfg.MOVINGMNIST.ILLUMINATION_LOWER,
                                   cfg.MOVINGMNIST.ILLUMINATION_UPPER))
    train_iter.load(os.path.join(cfg.MOVINGMNIST.MNIST_PATH, "mnistplus_train_30_10000.npz"))
    valid_iter = MovingMNISTAdvancedIterator(
        distractor_num=cfg.MOVINGMNIST.DISTRACTOR_NUM,
        initial_velocity_range=(cfg.MOVINGMNIST.VELOCITY_LOWER,
                                cfg.MOVINGMNIST.VELOCITY_UPPER),
        rotation_angle_range=(cfg.MOVINGMNIST.ROTATION_LOWER,
                              cfg.MOVINGMNIST.ROTATION_UPPER),
        scale_variation_range=(cfg.MOVINGMNIST.SCALE_VARIATION_LOWER,
                               cfg.MOVINGMNIST.SCALE_VARIATION_UPPER),
        illumination_factor_range=(cfg.MOVINGMNIST.ILLUMINATION_LOWER,
                                   cfg.MOVINGMNIST.ILLUMINATION_UPPER))
elif cfg.MODEL.DATASET == "KTH":
    from datasets.kth_iterator import parse_datasets
    data_iter = parse_datasets(device=cfg.GLOBAL.DEVICE,
                              batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                              sample_size=cfg.KTH.TOTAL_LEN)

    pass
else:
    print("This dataset has not implemented yet!!!")
## 3) Loss function
from operators.loss_funcs import MyMseMaeSSIM
criterion = MyMseMaeSSIM(seq_len=cfg.MODEL.IN_LEN + cfg.MODEL.OUT_LEN).to(cfg.GLOBAL.DEVICE)
## 4) Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.MODEL.LR )
## 5) Evaluator
from operators.mnist_evaluator import MNISTEvaluation
evaluater = MNISTEvaluation(seq_len=cfg.MODEL.OUT_LEN)
## 6) Training, validation, testing
if cfg.MODEL.DATASET == "MovingMNIST++" or cfg.MODEL.DATASET == "MovingMNIST":
    if cfg.MODEL.MODE == "train":
        train_mnist(model=model,
              train_set=train_iter,
              valid_set=valid_iter,
              irr_mode=cfg.MODEL.IRR_MODE,
              criterion=criterion,
              optimizer=optimizer,
              evaluator=evaluater)
    else:
        test_mnist(model=model,
             irr_mode=cfg.MODEL.IRR_MODE,
             evaluator=evaluater,
             criterion=criterion)
elif cfg.MODEL.DATASET == "KTH":
    if cfg.MODEL.MODE == "train":
        train_kth(model=model,
              dataset=data_iter,
              irr_mode=cfg.MODEL.IRR_MODE,
              criterion=criterion,
              optimizer=optimizer,
              evaluator=evaluater)
    else:
        test_kth(model=model,
             irr_mode=cfg.MODEL.IRR_MODE,
             evaluator=evaluater,
             criterion=criterion)
