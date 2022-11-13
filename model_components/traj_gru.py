import torch
from torch import nn
from configuration.config import cfg
import torch.nn.functional as F
from model_components.base_conv_rnn import BaseConvRNN
from torchdiffeq import odeint

# input: B, C, H, W
# flow: [B, 2, H, W]
def wrap(input, flow):
    B, C, H, W = input.size()
    # mesh grid
    # xx, yy: tensor (64,64)
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(cfg.GLOBAL.DEVICE)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(cfg.GLOBAL.DEVICE)
    # xx, yy: tensor (4,1,64,64)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    # grid (4,2,64,64)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid)
    return output

class TrajGRU_EN(BaseConvRNN):
    ### This class is used for computing one timestep
    def __init__(self, input_channel, num_filter, b_h_w, zoneout=0.0,
                 i2h_kernel=3, i2h_stride=1, i2h_pad=1,
                 h2h_kernel=5, h2h_stride=1, h2h_pad=1, h2h_dilate=1, L=None,
                 act_type=cfg.MODEL.RNN_ACT_TYPE, first_layer=True, last_layer=False,
                 attention=True, self_attention=False, ode_flag=False, zz_connection=False, device=None):
        super(TrajGRU_EN, self).__init__(num_filter=num_filter,
                                         b_h_w=b_h_w,
                                         h2h_kernel=h2h_kernel,
                                         h2h_dilate=h2h_dilate,
                                         i2h_kernel=i2h_kernel,
                                         i2h_pad=i2h_pad,
                                         i2h_stride=i2h_stride,
                                         act_type=act_type,
                                         prefix='TrajGRU_EN')
        super(TrajGRU_EN, self).__init__(num_filter=num_filter,
                                      b_h_w=b_h_w,
                                      h2h_kernel=h2h_kernel,
                                      h2h_dilate=h2h_dilate,
                                      i2h_kernel=i2h_kernel,
                                      i2h_pad=i2h_pad,
                                      i2h_stride=i2h_stride,
                                      act_type=act_type,
                                      prefix='TrajGRU_EN')
        self._L = L
        self.self_attention = self_attention
        self._zoneout = zoneout
        self.first_layer = first_layer
        self.last_layer = last_layer
        # Input-to-state transition: Wxz, Wxr, Wxh
        # reset_gate, update_gate, new_mem
        # out = num_filter*3
        self.i2h = nn.Conv2d(in_channels=input_channel,
                            out_channels=self._num_filter*3,
                            kernel_size=self._i2h_kernel,
                            stride=self._i2h_stride,
                            padding=self._i2h_pad,
                            dilation=self._i2h_dilate)

        # Input-to-flow transition
        self.i2f_conv1 = nn.Conv2d(in_channels=input_channel,
                                out_channels=num_filter, #32
                                kernel_size=(5, 5),
                                stride=1,
                                padding=(2, 2),
                                dilation=(1, 1))

        # hidden-to-flow transition
        # in = num_filter
        self.h2f_conv1 = nn.Conv2d(in_channels=self._num_filter,
                                   out_channels=num_filter, #32
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2),
                                   dilation=(1, 1))

        # generate flow
        self.flows_conv = nn.Conv2d(in_channels=num_filter, #32
                                   out_channels=self._L * 2,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2))

        # hidden state-to-state transition: Whh, Whz, Whr
        self.ret = nn.Conv2d(in_channels=self._num_filter*self._L,
                                   out_channels=self._num_filter*3,
                                   kernel_size=(1, 1),
                                   stride=1)

    # Input: B x C x H x W
    def _flow_generator(self, inputs, states):
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)
        else:
            i2f_conv1 = None
        h2f_conv1 = self.h2f_conv1(states)
        # self._act_type = torch.tanh
        ### Notice the size of the tensors
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self._act_type(f_conv1) # leaky relu
        flows = self.flows_conv(f_conv1) # B x 2L x H x W
        flows = torch.split(flows, 2, dim=1) # list: L: B x 2 x H x W
        return flows

    def forward(self, timestep_index, input, state, ode_states, zz_state, timestep):
        '''
        :param timestep: int
        :param input: B x C x W x H
        :param state: B x C x W x H
        :return:
        '''
        if state is None: # First cell h = 0
            state = torch.zeros((input.size(0), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
        #assert input == None, "At the encoding network, the input is None!"
        ## 1) Input convolution
        #B, C, H, W = input.size()
        i2h = self.i2h(input) # Bx3CxHxW
        i2h_slice = torch.split(i2h, self._num_filter, dim=1) # 3[BxCxHxW]
        # Flow generator
        flows =self._flow_generator(input, state)
        wrapped_data = []
        for j in range(len(flows)):
            flow = flows[j]
            wrapped_data.append(wrap(state, -flow))
        ## 2) State convolution
        wrapped_data = torch.cat(wrapped_data, dim=1)
        h2h = self.ret(wrapped_data)
        h2h_slice = torch.split(h2h, self._num_filter, dim=1)
        ## 3) TrajGRU operations
        reset_gate = torch.sigmoid(i2h_slice[0] + h2h_slice[0])
        update_gate = torch.sigmoid(i2h_slice[1] + h2h_slice[1])
        new_mem = torch.tanh(i2h_slice[2] + reset_gate * h2h_slice[2])
        new_state = update_gate * state + (1 - update_gate) * new_mem
        ## 4) Generate output of current cell
        return new_state

class TrajGRU_DE(BaseConvRNN):
    # b_h_w: input feature map size
    def __init__(self, input_channel, num_filter, b_h_w, zoneout=0.0,
                 i2h_kernel=3, i2h_stride=1, i2h_pad=1,
                 h2h_kernel=5, h2h_stride=1, h2h_pad=1, h2h_dilate=1, L=None,
                 act_type=cfg.MODEL.RNN_ACT_TYPE, first_layer=True, last_layer=False,
                 attention=True, self_attention=False, ode_flag=False, zz_connection=False, device=None):
        super(TrajGRU_DE, self).__init__(num_filter=num_filter,
                                      b_h_w=b_h_w,
                                      h2h_kernel=h2h_kernel,
                                      h2h_dilate=h2h_dilate,
                                      i2h_kernel=i2h_kernel,
                                      i2h_pad=i2h_pad,
                                      i2h_stride=i2h_stride,
                                      act_type=act_type,
                                      prefix='TrajGRU_DE')

        self.first_layer = first_layer
        self.last_layer = last_layer
        self._L = L
        self._zoneout = zoneout

        # Input-to-state transition: Wxz, Wxr, Wxh
        # reset_gate, update_gate, new_mem
        # out = num_filter*3
        self.i2h = nn.Conv2d(in_channels=input_channel,
                                 out_channels=self._num_filter * 3,
                                 kernel_size=self._i2h_kernel,
                                 stride=self._i2h_stride,
                                 padding=self._i2h_pad,
                                 dilation=self._i2h_dilate)

        # Input-to-flow transition
        self.i2f_conv1 = nn.Conv2d(in_channels=input_channel,
                                out_channels=num_filter, #32
                                kernel_size=(5, 5),
                                stride=1,
                                padding=(2, 2),
                                dilation=(1, 1))

        # hidden-to-flow transition
        # in = num_filter
        self.h2f_conv1 = nn.Conv2d(in_channels=self._num_filter,
                                   out_channels=num_filter, #32
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2),
                                   dilation=(1, 1))

        # generate flow
        self.flows_conv = nn.Conv2d(in_channels=num_filter, #32
                                   out_channels=self._L * 2,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2))

        # hidden state-to-state transition: Whh, Whz, Whr
        self.ret = nn.Conv2d(in_channels=self._num_filter*self._L,
                                   out_channels=self._num_filter*3,
                                   kernel_size=(1, 1),
                                   stride=1)

    # Input: B x C x H x W
    def _flow_generator(self, inputs, states):
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)
        else:
            i2f_conv1 = None
        h2f_conv1 = self.h2f_conv1(states)
        # self._act_type = torch.tanh
        ### Notice the size of the tensors
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self._act_type(f_conv1) # leaky relu
        flows = self.flows_conv(f_conv1) # B x 2L x H x W
        flows = torch.split(flows, 2, dim=1) # list: L: B x 2 x H x W
        return flows

    def forward(self, timestep_index, input, state, ode_states, zz_state, timestep):
        '''
        :param timestep: int
        :param input: B x C x W x H
        :param state: B x C x W x H
        :return:
        '''
        ## 1) Input convolution
        if input is None: # The first layer of forecasting network
            i2h_slice = None
        else:
            i2h = self.i2h(input) # Bx3CxHxW
            i2h_slice = torch.split(i2h, self._num_filter, dim=1)
        #assert state == None, "At the forecasting network, the state is None!"

        # Flow generator
        flows =self._flow_generator(input, state)
        wrapped_data = []
        for j in range(len(flows)):
            flow = flows[j]
            wrapped_data.append(wrap(state, -flow))
        ## 2) State convolution
        wrapped_data = torch.cat(wrapped_data, dim=1)
        h2h = self.ret(wrapped_data)
        h2h_slice = torch.split(h2h, self._num_filter, dim=1)
        ## 3) TrajGRU operations
        if i2h_slice is not None:
            reset_gate = torch.sigmoid(i2h_slice[0] + h2h_slice[0])
            update_gate = torch.sigmoid(i2h_slice[1] + h2h_slice[1])
            new_mem = torch.tanh(i2h_slice[2] + reset_gate * h2h_slice[2])
        else:
            reset_gate = torch.sigmoid(h2h_slice[0])
            update_gate = torch.sigmoid(h2h_slice[1])
            new_mem = torch.tanh(reset_gate * h2h_slice[2])
        new_state = update_gate * state + (1 - update_gate) * new_mem
        ## 4) Generate output of current cell
        return new_state