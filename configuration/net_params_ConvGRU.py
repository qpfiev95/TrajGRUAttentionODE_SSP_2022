import sys
sys.path.insert(0, '..')
from model_components.conv_gru import ConvGRU_EN, ConvGRU_DE
from collections import OrderedDict
from configuration.config import cfg

# parameters
batch_size = cfg.MODEL.BATCH_SIZE
init_channel = cfg.MODEL.INIT_CHANNEL
init_size = cfg.MODEL.INIT_SIZE
encoder_params = [
    [
# in_channel, out_channel, kernel_size, stride, padding
        OrderedDict({'conv1_leaky_1': [init_channel, 64, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 96, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 128, 3, 2, 1]}),
    ],
    [
        ConvGRU_EN(input_channel=64, num_filter=64, b_h_w=(batch_size, init_size, init_size), zoneout=0.0,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_stride=(1, 1), h2h_pad=(1, 1), h2h_dilate=(1, 1), L=None,
                act_type=cfg.MODEL.RNN_ACT_TYPE,
                first_layer=True, last_layer=False, self_attention=False, ode_flag=False, zz_connection=False),

        ConvGRU_EN(input_channel=96, num_filter=96, b_h_w=(batch_size, init_size//2, init_size//2), zoneout=0.0,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_stride=(1, 1), h2h_pad=(1, 1), h2h_dilate=(1, 1), L=None,
                act_type=cfg.MODEL.RNN_ACT_TYPE,
                first_layer=False, last_layer=False, self_attention=False, ode_flag=False, zz_connection=False),

        ConvGRU_EN(input_channel=128, num_filter=128, b_h_w=(batch_size, init_size//4, init_size//4), zoneout=0.0,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_stride=(1, 1), h2h_pad=(1, 1), h2h_dilate=(1, 1), L=None,
                act_type=cfg.MODEL.RNN_ACT_TYPE,
                first_layer=False, last_layer=True, self_attention=False, ode_flag=False, zz_connection=False)
    ]
]
########################################################################
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
        ConvGRU_DE(input_channel=128, num_filter=128, b_h_w=(batch_size, init_size//4, init_size//4), zoneout=0.0,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_stride=(1, 1), h2h_pad=(1, 1), h2h_dilate=(1, 1), L=None,
                act_type=cfg.MODEL.RNN_ACT_TYPE,
                first_layer=True, last_layer=False, attention=False, ode_flag=False, zz_connection=False),

        ConvGRU_DE(input_channel=96, num_filter=96, b_h_w=(batch_size, init_size//2, init_size//2), zoneout=0.0,
                i2h_kernel=(3,3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3,3), h2h_stride=(1, 1), h2h_pad=(1, 1), h2h_dilate=(1, 1), L=None,
                act_type=cfg.MODEL.RNN_ACT_TYPE,
                first_layer=False, last_layer=False, attention=False, ode_flag=False, zz_connection=False),

        ConvGRU_DE(input_channel=64, num_filter=64, b_h_w=(batch_size, init_size, init_size), zoneout=0.0,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_stride=(1, 1), h2h_pad=(1, 1), h2h_dilate=(1, 1), L=None,
                act_type=cfg.MODEL.RNN_ACT_TYPE,
                first_layer=False, last_layer=True, attention=False, ode_flag=False, zz_connection=False)
    ]
]

