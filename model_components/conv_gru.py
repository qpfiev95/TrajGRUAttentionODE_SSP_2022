import torch
from torch import nn
from configuration.config import cfg
import torch.nn.functional as F
from model_components.base_conv_rnn import BaseConvRNN

class ConvGRU_EN(BaseConvRNN):
    ### This class is used for computing one timestep
    def __init__(self, input_channel, num_filter, b_h_w, zoneout=0.0,
                 i2h_kernel=3, i2h_stride=1, i2h_pad=1,
                 h2h_kernel=5, h2h_stride=1, h2h_pad=1, h2h_dilate=1, L=None,
                 act_type=cfg.MODEL.RNN_ACT_TYPE, first_layer=True, last_layer=False,
                 self_attention=True, ode_flag=False, zz_connection=False, device=None):
        super(ConvGRU_EN, self).__init__(num_filter=num_filter,
                                         b_h_w=b_h_w,
                                         h2h_kernel=h2h_kernel,
                                         h2h_dilate=h2h_dilate,
                                         i2h_kernel=i2h_kernel,
                                         i2h_pad=i2h_pad,
                                         i2h_stride=i2h_stride,
                                         act_type=act_type,
                                         prefix='ConvGRU_EN')
        self.device = device
        self.zz_connection = zz_connection
        self.ode_flag = ode_flag
        self.self_attention = self_attention
        self._zoneout = zoneout
        self.first_layer = first_layer
        self.last_layer = last_layer
        # Trainable parameters:
        self.i2h_conv = nn.Conv2d(in_channels=input_channel,
                                  out_channels=num_filter * 3,  # 32
                                  kernel_size=i2h_kernel,
                                  stride=i2h_stride,
                                  padding=i2h_pad)

        self.h2h_conv = nn.Conv2d(in_channels=input_channel,
                                  out_channels=num_filter * 3,  # 32
                                  kernel_size=i2h_kernel,
                                  stride=i2h_stride,
                                  padding=i2h_pad)

    def forward(self, timestep_index, inputs, states, ode_states, zz_state, timestep):
        '''
        :param timestep: int
        :param inputs: S x B x C x W x H
        :param states: S - 1 x B x C x W x H
        :return:
        new_state: B x C x W x H
        new_A_state: B x C x W x H
        '''
        if states is None:  # At the first timestep
            if self.device is None:
                state = torch.zeros((inputs[-1].size(0), self._num_filter, self._state_height,
                                     self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
            else:
                state = torch.zeros((inputs[-1].size(0), self._num_filter, self._state_height,
                                     self._state_width), dtype=torch.float).to(self.device)
        else:
            state = states[-1]
        ## 1) Input convolution
        if inputs is not None:
            recent_input = inputs[-1]
            input_conv = self.i2h_conv(recent_input)  # Bx6CxHxW
            input_conv_slice = torch.split(input_conv, self._num_filter, dim=1)  # 6[BxCxHxW]
        ## 2) State convolution
        state_conv = self.h2h_conv(state)  # Bx3CxHxW
        state_conv_slice = torch.split(state_conv, self._num_filter, dim=1)
        ## 3) ConvGRU operations
        if self.self_attention:
            reset_gate = torch.sigmoid(input_conv_slice[0] + state_conv_slice[0])
            update_gate = torch.sigmoid(input_conv_slice[1] + state_conv_slice[1])
            new_mem = torch.tanh(input_conv_slice[2] + reset_gate * state_conv_slice[2])
            new_state = update_gate * state + (1 - update_gate) * new_mem
            new_a_state = new_state
        else:
            reset_gate = torch.sigmoid(input_conv_slice[0] + state_conv_slice[0])
            update_gate = torch.sigmoid(input_conv_slice[1] + state_conv_slice[1])
            new_mem = torch.tanh(input_conv_slice[2] + reset_gate * state_conv_slice[2])
            new_state = update_gate * state + (1 - update_gate) * new_mem
            new_a_state = new_state
        return new_state, new_a_state

class ConvGRU_DE(BaseConvRNN):
    ### This class is used for computing one timestep
    def __init__(self, input_channel, num_filter, b_h_w, zoneout=0.0,
                 i2h_kernel=3, i2h_stride=1, i2h_pad=1,
                 h2h_kernel=5, h2h_stride=1, h2h_pad=1, h2h_dilate=1, L=None,
                 act_type=cfg.MODEL.RNN_ACT_TYPE, first_layer=True, last_layer=False,
                 attention=True, self_attention=False, ode_flag=False, zz_connection=False, device=None):
        super(ConvGRU_DE, self).__init__(num_filter=num_filter,
                                         b_h_w=b_h_w,
                                         h2h_kernel=h2h_kernel,
                                         h2h_dilate=h2h_dilate,
                                         i2h_kernel=i2h_kernel,
                                         i2h_pad=i2h_pad,
                                         i2h_stride=i2h_stride,
                                         act_type=act_type,
                                         prefix='ConvGRU_FO')
        self.device = device
        self.zz_connection = zz_connection
        self.ode_flag = ode_flag
        self.attention = attention
        self.self_attention = self_attention
        self._zoneout = zoneout
        self.first_layer = first_layer
        self.last_layer = last_layer
        # Trainable parameters:
        self.i2h_conv = nn.Conv2d(in_channels=input_channel,
                                  out_channels=num_filter * 3,  # 32
                                  kernel_size=i2h_kernel,
                                  stride=i2h_stride,
                                  padding=i2h_pad)
        self.h2h_conv = nn.Conv2d(in_channels=input_channel,
                                  out_channels=num_filter * 3,  # 32
                                  kernel_size=h2h_kernel,
                                  stride=h2h_stride,
                                  padding=h2h_pad)

    def forward(self, timestep_index, inputs, states, ode_states, zz_state, timestep):
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
        if states is None:  # At the first timestep
            if self.device is None:
                state = torch.zeros((inputs[-1].size(0), self._num_filter, self._state_height,
                                     self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
            else:
                state = torch.zeros((inputs[-1].size(0), self._num_filter, self._state_height,
                                     self._state_width), dtype=torch.float).to(self.device)
        elif self.zz_connection:
            if self.ode_flag:
                state = self.combined_state_generator(states[-1], zz_state, ode_states[timestep_index + 1])
            else:
                state = self.combined_state_generator(states[-1], zz_state, None)
        else:
            state = states[-1]
        ## 1) Input convolution
        # B, C, H, W = input.size()
        if not self.zz_connection:
            if self.attention:
                recent_input = inputs[-1]
                input_conv = self.i2h_conv(recent_input)
                input_conv_slice = torch.split(input_conv, self._num_filter, dim=1)  # 5[BxCxHxW]
            elif inputs is not None:
                recent_input = inputs[-1]
                input_conv = self.i2h_conv(recent_input)  # Bx6CxHxW
                input_conv_slice = torch.split(input_conv, self._num_filter, dim=1)  # 5[BxCxHxW]
            else:
                recent_input = None
        else:
            recent_input = None
        ## 2) State convolution
        state_conv = self.h2h_conv(state)  # Bx3CxHxW
        state_conv_slice = torch.split(state_conv, self._num_filter, dim=1)
        ## 3) ConvGRU operations
        if recent_input is not None:
            reset_gate = torch.sigmoid(input_conv_slice[0] + state_conv_slice[0])
            update_gate = torch.sigmoid(input_conv_slice[1] + state_conv_slice[1])
            new_mem = torch.tanh(input_conv_slice[2] + reset_gate * state_conv_slice[2])
            new_state = update_gate * state + (1 - update_gate) * new_mem
            new_a_state = new_state
        else:
            reset_gate = torch.sigmoid(state_conv_slice[0])
            update_gate = torch.sigmoid(state_conv_slice[1])
            new_mem = torch.tanh(reset_gate * state_conv_slice[2])
            new_state = update_gate * state + (1 - update_gate) * new_mem
            new_a_state = new_state
        return new_state, new_a_state