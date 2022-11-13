import sys
sys.path.insert(0, '..')
from model_components.traj_gru_attention_ode import TrajGRU_Attention_ODE_EN, TrajGRU_Attention_ODE_DE
from collections import OrderedDict
from configuration.make_layers import make_convnet
from configuration.config import cfg
from model_components.ode_func import DiffeqSolver, ODEFunc

### Params for the model and dataset
batch_size = cfg.MODEL.TRAIN.BATCH_SIZE
init_channel = cfg.MODEL.INIT_CHANNEL
init_size = cfg.MODEL.INIT_SIZE
###
encoder_params = [
    [
# in_channel, out_channel, kernel_size, stride, padding
        OrderedDict({'conv1_leaky_1': [init_channel, 64, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 96, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 128, 3, 2, 1]}),
    ],

    [
        TrajGRU_Attention_ODE_EN(input_channel=64, num_filter=64, b_h_w=(batch_size, init_size, init_size), zoneout=0.0,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_stride=(1, 1), h2h_pad=(1, 1), h2h_dilate=(1, 1), L=9,
                act_type=cfg.MODEL.RNN_ACT_TYPE,
                first_layer=False, last_layer=False, self_attention=False, ode_flag=False, zz_connection=True, device=cfg.GLOBAL.DEVICE_EN),

        TrajGRU_Attention_ODE_EN(input_channel=96, num_filter=96, b_h_w=(batch_size, init_size//2, init_size//2), zoneout=0.0,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_stride=(1, 1), h2h_pad=(1, 1), h2h_dilate=(1, 1), L=9,
                act_type=cfg.MODEL.RNN_ACT_TYPE,
                first_layer=False, last_layer=False, self_attention=True, ode_flag=False, zz_connection=False, device=cfg.GLOBAL.DEVICE_EN),

        TrajGRU_Attention_ODE_EN(input_channel=128, num_filter=128, b_h_w=(batch_size, init_size//4, init_size//4), zoneout=0.0,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_stride=(1, 1), h2h_pad=(1, 1), h2h_dilate=(1, 1), L=9,
                act_type=cfg.MODEL.RNN_ACT_TYPE,
                first_layer=False, last_layer=True, self_attention=True, ode_flag=False, zz_connection=False, device=cfg.GLOBAL.DEVICE_EN)
    ],
    [
        # in_channel, out_channel, kernel_size, stride, padding
        # Ho = (Hi-1)*s - 2*p + d*(k-1) + o_p + 1
        OrderedDict({'deconv_leaky': [128, 64, 4, 4, 0]}),
        OrderedDict({'conv_leaky': [64, 128, 4, 4, 1]})
    ]
]
########################################################################
layer_num = 1  # en_a23_fo_a21_odenet_x2: 2 ||| others: 1
unit_num = 48 #   en_a23_fo_a21_odenet_x2: 64 ||| others: 48

forecaster_ode_net_1 = make_convnet(n_inputs=128, n_outputs=128,
                               n_layers=layer_num , n_units=unit_num).to(cfg.GLOBAL.DEVICE_EN)
forecaster_ode_func_1 = ODEFunc(input_dim=128, latent_dim=128,
                           ode_func_net=forecaster_ode_net_1)
forecaster_ode_solver_1 = DiffeqSolver(input_dim=128, ode_func=forecaster_ode_func_1,
                                  method=cfg.MODEL.ODE_METHODS,
                                  latents=128,
                                  odeint_rtol=1e-4, odeint_atol=1e-5, device=cfg.GLOBAL.DEVICE_EN)

forecaster_ode_net_2 = make_convnet(n_inputs=96, n_outputs=96,
                               n_layers=layer_num , n_units=unit_num).to(cfg.GLOBAL.DEVICE_EN)
forecaster_ode_func_2 = ODEFunc(input_dim=96, latent_dim=96,
                           ode_func_net=forecaster_ode_net_2)
forecaster_ode_solver_2 = DiffeqSolver(input_dim=96, ode_func=forecaster_ode_func_2,
                                  method=cfg.MODEL.ODE_METHODS,
                                  latents=96,
                                  odeint_rtol=1e-4, odeint_atol=1e-5, device=cfg.GLOBAL.DEVICE_EN)
forecaster_ode_net_3 = make_convnet(n_inputs=64, n_outputs=64,
                               n_layers=layer_num , n_units=unit_num).to(cfg.GLOBAL.DEVICE_EN)
forecaster_ode_func_3 = ODEFunc(input_dim=64, latent_dim=64,
                           ode_func_net=forecaster_ode_net_3)
forecaster_ode_solver_3 = DiffeqSolver(input_dim=64, ode_func=forecaster_ode_func_3,
                                  method=cfg.MODEL.ODE_METHODS,
                                  latents=64,
                                  odeint_rtol=1e-4, odeint_atol=1e-5, device=cfg.GLOBAL.DEVICE_EN)

################################################
decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [128, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 64, 4, 2, 1]}),
        OrderedDict({'deconv3_leaky_1': [64, 32, 3, 1, 1]}),
        OrderedDict({
            'conv4_leaky': [32, 16, 3, 1, 1],
            'conv4': [16, init_channel, 1, 1, 0]
        }),
    ],

    [
        TrajGRU_Attention_ODE_DE(input_channel=128, num_filter=128, b_h_w=(batch_size, init_size//4, init_size//4), zoneout=0.0,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_stride=(1, 1), h2h_pad=(1, 1), h2h_dilate=(1, 1), L=9,
                act_type=cfg.MODEL.RNN_ACT_TYPE,
                first_layer=False, last_layer=False, attention=False, ode_flag=False, zz_connection=True, device=cfg.GLOBAL.DEVICE_DE),

        TrajGRU_Attention_ODE_DE(input_channel=96, num_filter=96, b_h_w=(batch_size, init_size//2, init_size//2), zoneout=0.0,
                i2h_kernel=(3,3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3,3), h2h_stride=(1, 1), h2h_pad=(1, 1), h2h_dilate=(1, 1), L=9,
                act_type=cfg.MODEL.RNN_ACT_TYPE,
                first_layer=False, last_layer=False, attention=True, ode_flag=False, zz_connection=False, device=cfg.GLOBAL.DEVICE_DE),

        TrajGRU_Attention_ODE_DE(input_channel=64, num_filter=64, b_h_w=(batch_size, init_size, init_size), zoneout=0.0,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_stride=(1, 1), h2h_pad=(1, 1), h2h_dilate=(1, 1), L=9,
                act_type=cfg.MODEL.RNN_ACT_TYPE,
                first_layer=False, last_layer=True, attention=True, ode_flag=False, zz_connection=False, device=cfg.GLOBAL.DEVICE_DE)
    ],
    [
        forecaster_ode_solver_1,
        forecaster_ode_solver_2,
        forecaster_ode_solver_3,
    ],
    [
        # in_channel, out_channel, kernel_size, stride, padding
        # Ho = (Hi-1)*s - 2*p + d*(k-1) + o_p + 1
        OrderedDict({'conv_leaky': [32, 128, 4, 4, 1]})
    ]
]

