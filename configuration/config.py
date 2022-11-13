from helpers.ordered_easydict import OrderedEasyDict as edict
import os
import torch
from operators.activation import activation
import numpy as np
#################################### GLOBAL VARIBLE ############################
__C = edict()
cfg = __C
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
__C.GLOBAL = edict()
__C.GLOBAL.DEVICE_MODE = "single"
__C.GLOBAL.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __C.GLOBAL.DEVICE_MODE == "single":
    __C.GLOBAL.DEVICE_EN = torch.device("cuda:0")
    __C.GLOBAL.DEVICE_DE = torch.device("cuda:0")
elif torch.cuda.device(1):
    __C.GLOBAL.DEVICE_EN = torch.device("cuda:0")
    __C.GLOBAL.DEVICE_DE = torch.device("cuda:1")
__C.GLOBAL.TOTAL_LEN = 10
################### MODEL ###################################
__C.MODEL = edict()
__C.MODEL.TECH = "TrajGRU-Attention"  # "ConvGRU" | "TrajGRU" | "TrajGRU-Attention" | "TrajGRU-Attention-ODE" | "Vid-ODE"
__C.MODEL.MODE = "train"  # "train"  | "test"
__C.MODEL.DATASET = "MovingMNIST" # "MovingMNIST" | "MovingMNIST++" | "KTH"
__C.MODEL.INIT_CHANNEL = 1 # MovingMNIST: 1, KTH: 3
__C.MODEL.INIT_SIZE = 64 # MovingMNIST: 64, KTH: 128
__C.MODEL.IN_LEN = 10
__C.MODEL.OUT_LEN = 10
__C.MODEL.TOTAL_LEN = 30 # 30: irr | 20: re
__C.MODEL.T_EN = np.arange(1.0,float(__C.MODEL.IN_LEN),1.0)
__C.MODEL.T_DE = np.arange(float(__C.MODEL.IN_LEN + 1.0),float(__C.MODEL.OUT_LEN),1.0)
__C.MODEL.ODE_METHODS = "euler" ### "dopri5" "euler"
__C.MODEL.IRR_MODE = "regular" #  "encoder" | "decoder"| "both": irregularly sampled at encoder and decoder| "regular"
__C.MODEL.RNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True) ### inplace = True, testing
## Train
__C.MODEL.TRAIN = edict()
__C.MODEL.TRAIN.NUM_EPOCHS = 20
__C.MODEL.TRAIN.MAX_ITER = 200000 # 200000
__C.MODEL.TRAIN.TEST_ITER_INTERVAL = 2000# 2000
__C.MODEL.TRAIN.VALID_ITER_INTERVAL = 8000 # 8000
__C.MODEL.TRAIN.UPDATE_ITER_INTERVAL = 8000 # 8000
__C.MODEL.TRAIN.PRINT_ITER_INTERVAL = 50 #50
__C.MODEL.TRAIN.SAVE_ITER = 8000 #8000
__C.MODEL.TRAIN.BATCH_SIZE = 4
__C.MODEL.DECAY_RATE = 0.99
__C.MODEL.LR = 1e-4 #  1e-4 | 1e-3 (Starting learning rate)
## Save
__C.MODEL.SAVE_CHECKPOINT_DIR = os.path.join(__C.ROOT_DIR,
                                             'results/save_kth_irr/en_re_fo_re/trajgru_attention_ode_v3/model_params/en_a23_fo_a21/')
__C.MODEL.LOAD_TEST_DIR = os.path.join(__C.ROOT_DIR,
                                             'results/save_kth_irr/en_re_fo_re/trajgru_attention_ode_v3/model_params/en_a23_fo_a21/')
__C.MODEL.TRAIN_IMAGE_PATH = os.path.join(__C.ROOT_DIR,
                                             'results/save_kth_irr/en_re_fo_re/trajgru_attention_ode_v3/model_params/en_a23_fo_a21/')
__C.MODEL.TEST_IMAGE_PATH = os.path.join(__C.ROOT_DIR,
                                             'results/save_kth_irr/en_re_fo_re/trajgru_attention_ode_v3/model_params/en_a23_fo_a21/')
__C.MODEL.TRAINING_MODE = "continue"
__C.MODEL.LOAD_TRAIN_CONTINUE_DIR = os.path.join(__C.ROOT_DIR,
                                             'results/save_kth_irr/en_re_fo_re/trajgru_attention_ode_v3/model_params/en_a23_fo_a21/')
## Validation
__C.MODEL.VALID = edict()
__C.MODEL.VALID.VALID_NUM = 500 # 500
## Validation
__C.MODEL.TEST = edict()
__C.MODEL.TEST.TEST_NUM = 1250
###     MOVING MNIST    #########################################################################
__C.MOVINGMNIST = edict()
__C.MOVINGMNIST.MNIST_PATH = os.path.join(__C.ROOT_DIR, 'datasets/mnist')
__C.MOVINGMNIST.MODEL_SAVE_DIR = "results/save"
__C.MOVINGMNIST.PLUS = True
## Test
__C.MOVINGMNIST.TEST = edict()
__C.MOVINGMNIST.TEST.TEST_FILE_FIXED = os.path.join(__C.MOVINGMNIST.MNIST_PATH, "test_mnist_data_20_2000.npy")
__C.MOVINGMNIST.TEST.TEST_FILE = os.path.join(__C.MOVINGMNIST.MNIST_PATH, "mnist_20_1000.npz")
__C.MOVINGMNIST.EXAMPLE = "./"
## Other parameters for MovingMNIST++
__C.MOVINGMNIST.DISTRACTOR_NUM = 0
__C.MOVINGMNIST.VELOCITY_LOWER = 0.0
__C.MOVINGMNIST.VELOCITY_UPPER = 3.6
__C.MOVINGMNIST.SCALE_VARIATION_LOWER = 1/1.1
__C.MOVINGMNIST.SCALE_VARIATION_UPPER = 1.1
__C.MOVINGMNIST.ROTATION_LOWER = -30
__C.MOVINGMNIST.ROTATION_UPPER = 30
__C.MOVINGMNIST.ILLUMINATION_LOWER = 0.6
__C.MOVINGMNIST.ILLUMINATION_UPPER = 1.0
__C.MOVINGMNIST.DIGIT_NUM = 3
__C.MOVINGMNIST.SUB_OUT_LEN = 2
__C.MOVINGMNIST.TESTING_LEN = 20
__C.MOVINGMNIST.IMG_SIZE = 64
###     KTH Action    #########################################################################
__C.KTH.TOTAL_LEN = 20

### Vid-ODE  ####################################################################################
__C.VidODE = edict()
__C.VidODE.INPUT_SIZE = 80 ### 64 or 80
__C.VidODE.INPUT_DIM = 1 # MovingMNIST: 1 | KTH Action: 3
__C.VidODE.INIT_DIM = 128 # 64, 128
__C.VidODE.NUM_DOWNSAMPLING = 2 # 3
__C.VidODE.NUM_LAYERS = 3 # 3, 4
__C.VidODE.RUN_BACKWARDS = True
__C.VidODE.LAMBDA_ADV = 0.003
__C.VidODE.LAMBDA_ADV = 0.005 # Adversarial Loss Lambda 0.007
__C.VidODE.INPUT_NORM = False
#### GAN
__C.VidODE.EXTRAPOLATION_MODE = True # => interpolation
__C.VidODE.IRREGULAR = False # Train with irregular time-step data
__C.VidODE.SAMPLE_SIZE = 20 # Number of time points to sub-sample
__C.VidODE.INPUT_DIM = 1 # 1 with MovingMNIST


