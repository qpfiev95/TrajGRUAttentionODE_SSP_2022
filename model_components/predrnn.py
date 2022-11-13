import torch
from torch import nn
from configuration.config import cfg
import torch.nn.functional as F
from model_components.base_conv_rnn import BaseConvRNN
from torchdiffeq import odeint


class ConvGRU_A_ZZmem_EN(BaseConvRNN):
    ### This class is used for computing one timestep
    def __init__(self, input_channel, num_filter, b_h_w, zoneout=0.0,
                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                 h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                 act_type=cfg.MODEL.RNN_ACT_TYPE, first_layer=True, last_layer=False,
                 self_attention=True):
        super(ConvGRU_A_ZZmem_EN, self).__init__(num_filter=num_filter,
                                      b_h_w=b_h_w,
                                      h2h_kernel=h2h_kernel,
                                      h2h_dilate=h2h_dilate,
                                      i2h_kernel=i2h_kernel,
                                      i2h_pad=i2h_pad,
                                      i2h_stride=i2h_stride,
                                      act_type=act_type,
                                      prefix='ConvGRU_A_ZZmem')
        self.self_attention = self_attention
        self._zoneout = zoneout
        self.first_layer = first_layer
        self.last_layer = last_layer
        # Trainable parameters:
        self.w11 = nn.Conv2d(in_channels=2*input_channel,
                                     out_channels=num_filter,
                                     kernel_size=1,
                                     stride=1)

        self.w_x = nn.Conv2d(in_channels=input_channel,
                            out_channels=num_filter*7,
                            kernel_size=self._i2h_kernel,
                            stride=self._i2h_stride,
                            padding=self._i2h_pad,
                            dilation=self._i2h_dilate)

        self.w_h = nn.Conv2d(in_channels=input_channel,
                            out_channels=num_filter*4,
                            kernel_size=self._i2h_kernel,
                            stride=self._i2h_stride,
                            padding=self._i2h_pad,
                            dilation=self._i2h_dilate)

        self.w_m = nn.Conv2d(in_channels=input_channel,
                            out_channels=num_filter*3,
                            kernel_size=self._i2h_kernel,
                            stride=self._i2h_stride,
                            padding=self._i2h_pad,
                            dilation=self._i2h_dilate)
        self.w_m_new = nn.Conv2d(in_channels=input_channel,
                             out_channels=num_filter,
                             kernel_size=self._i2h_kernel,
                             stride=self._i2h_stride,
                             padding=self._i2h_pad,
                             dilation=self._i2h_dilate)
        self.w_c_new = nn.Conv2d(in_channels=input_channel,
                             out_channels=num_filter,
                             kernel_size=self._i2h_kernel,
                             stride=self._i2h_stride,
                             padding=self._i2h_pad,
                             dilation=self._i2h_dilate)

    def forward(self, timestep, inputs, states, C_state, M_state):
        '''
        :param timestep: int
        :param inputs: S x B x C x W x H
        :param states: S - 1 x B x C x W x H
        :return:
        new_state: B x C x W x H
        new_A_state: B x C x W x H
        new_M_state: B x C x W x H
        '''
        if states is None: # At the first timestep
            state = torch.zeros((inputs[-1].size(0), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
            M_state = torch.zeros((inputs[-1].size(0), self._num_filter, self._state_height,
                                   self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
            C_state = torch.zeros((inputs[-1].size(0), self._num_filter, self._state_height,
                                   self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
        else:
            state = states[-1]
        #assert input == None, "At the encoding network, the input is None!"
        ## 1) Input convolution
        #B, C, H, W = input.size()
        if inputs is not None:
            input_conv = self.w_x(inputs[-1]) # Bx 7C x H x W
            input_conv_slice = torch.split(input_conv, self._num_filter, dim=1) # 7[BxCxHxW]
        ## 2) State convolution
        state_conv = self.w_h(state) # Bx 4C x H x W
        state_conv_slice = torch.split(state_conv, self._num_filter, dim=1)  # 4[BxCxHxW]
        M_state_conv = self.w_m(M_state)
        M_state_conv_slice = torch.split(M_state_conv, self._num_filter, dim=1) # 3[BxCxHxW]
        ## 3) predrnn operations
        g_h = torch.tanh(input_conv_slice[0] + state_conv_slice[0])
        i_h = torch.sigmoid(input_conv_slice[1] + state_conv_slice[1])
        f_h = torch.sigmoid(input_conv_slice[2] + state_conv_slice[2])
        c = f_h * C_state + i_h * g_h
        g_m = torch.tanh(input_conv_slice[3] + M_state_conv_slice[0])
        i_m = torch.sigmoid(input_conv_slice[4] + M_state_conv_slice[1])
        f_m = torch.sigmoid(input_conv_slice[5] + M_state_conv_slice[2])
        m = f_m * M_state + i_m * g_m
        o = torch.sigmoid(input_conv_slice[6] + state_conv_slice[3] +
                          self.w_m_new(m) + self.w_c_new(c))
        new_state = o * torch.tanh(self.w11(torch.cat([c,m], dim=1)))

        return new_state, c, m

class ConvGRU_A_ZZmem_FO(BaseConvRNN):
    ### This class is used for computing one timestep
    def __init__(self, input_channel, num_filter, b_h_w, zoneout=0.0,
                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                 h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                 act_type=cfg.MODEL.RNN_ACT_TYPE, first_layer=True, last_layer=False,
                 self_attention=True):
        super(ConvGRU_A_ZZmem_FO, self).__init__(num_filter=num_filter,
                                      b_h_w=b_h_w,
                                      h2h_kernel=h2h_kernel,
                                      h2h_dilate=h2h_dilate,
                                      i2h_kernel=i2h_kernel,
                                      i2h_pad=i2h_pad,
                                      i2h_stride=i2h_stride,
                                      act_type=act_type,
                                      prefix='ConvGRU_A_ZZmem')
        self.self_attention = self_attention
        self._zoneout = zoneout
        self.first_layer = first_layer
        self.last_layer = last_layer
        # Trainable parameters:
        self.w11 = nn.Conv2d(in_channels=2*input_channel,
                                     out_channels=num_filter,
                                     kernel_size=1,
                                     stride=1)

        self.w_x = nn.Conv2d(in_channels=input_channel,
                            out_channels=num_filter*7,
                            kernel_size=self._i2h_kernel,
                            stride=self._i2h_stride,
                            padding=self._i2h_pad,
                            dilation=self._i2h_dilate)

        self.w_h = nn.Conv2d(in_channels=input_channel,
                            out_channels=num_filter*4,
                            kernel_size=self._i2h_kernel,
                            stride=self._i2h_stride,
                            padding=self._i2h_pad,
                            dilation=self._i2h_dilate)

        self.w_m = nn.Conv2d(in_channels=input_channel,
                            out_channels=num_filter*3,
                            kernel_size=self._i2h_kernel,
                            stride=self._i2h_stride,
                            padding=self._i2h_pad,
                            dilation=self._i2h_dilate)
        self.w_m_new = nn.Conv2d(in_channels=input_channel,
                             out_channels=num_filter,
                             kernel_size=self._i2h_kernel,
                             stride=self._i2h_stride,
                             padding=self._i2h_pad,
                             dilation=self._i2h_dilate)
        self.w_c_new = nn.Conv2d(in_channels=input_channel,
                             out_channels=num_filter,
                             kernel_size=self._i2h_kernel,
                             stride=self._i2h_stride,
                             padding=self._i2h_pad,
                             dilation=self._i2h_dilate)

    def forward(self, timestep, inputs, states, C_state, M_state):
        '''
        :param timestep: int
        :param inputs: S x B x C x W x H
        :param states: S x B x C x W x H
        :params A_state: B x C x W x H
        :params M_state: B x C x W x H
        :return:
        new_state: B x C x W x H
        new_A_state: B x C x W x H
        new_M_state: B x C x W x H
        '''
        if states is None: # At the first timestep
            state = torch.zeros((inputs[-1].size(0), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
            M_state = torch.zeros((inputs[-1].size(0), self._num_filter, self._state_height,
                                   self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
            C_state = torch.zeros((inputs[-1].size(0), self._num_filter, self._state_height,
                                   self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
        else:
            state = states[-1]
        #assert input == None, "At the encoding network, the input is None!"
        ## 1) Input convolution
        #B, C, H, W = input.size()
        if inputs is not None:
            input_conv = self.w_x(inputs[-1]) # Bx 7C x H x W
            input_conv_slice = torch.split(input_conv, self._num_filter, dim=1) # 7[BxCxHxW]
        ## 2) State convolution
        state_conv = self.w_h(state) # Bx 4C x H x W
        state_conv_slice = torch.split(state_conv, self._num_filter, dim=1)  # 4[BxCxHxW]
        M_state_conv = self.w_m(M_state)
        M_state_conv_slice = torch.split(M_state_conv, self._num_filter, dim=1) # 3[BxCxHxW]
        ## 3) ConvGRU operations
        if inputs is None: # Case 1: At the bottom layer of the forecaster
            g_h = torch.tanh(state_conv_slice[0])
            i_h = torch.sigmoid(state_conv_slice[1])
            f_h = torch.sigmoid(state_conv_slice[2])
            c = f_h * C_state + i_h * g_h
            g_m = torch.tanh(M_state_conv_slice[0])
            i_m = torch.sigmoid(M_state_conv_slice[1])
            f_m = torch.sigmoid(M_state_conv_slice[2])
            m = f_m * M_state + i_m * g_m
            o = torch.sigmoid(state_conv_slice[3] +
                              self.w_m_new(m) + self.w_c_new(c))
            new_state = o * torch.tanh(self.w11(torch.cat([c, m], dim=1)))
        else:
            g_h = torch.tanh(input_conv_slice[0] + state_conv_slice[0])
            i_h = torch.sigmoid(input_conv_slice[1] + state_conv_slice[1])
            f_h = torch.sigmoid(input_conv_slice[2] + state_conv_slice[2])
            c = f_h * C_state + i_h * g_h
            g_m = torch.tanh(input_conv_slice[3] + M_state_conv_slice[0])
            i_m = torch.sigmoid(input_conv_slice[4] + M_state_conv_slice[1])
            f_m = torch.sigmoid(input_conv_slice[5] + M_state_conv_slice[2])
            m = f_m * M_state + i_m * g_m
            o = torch.sigmoid(input_conv_slice[6] + state_conv_slice[3] +
                              self.w_m_new(m) + self.w_c_new(c))
            new_state = o * torch.tanh(self.w11(torch.cat([c, m], dim=1)))

        return new_state, c, m