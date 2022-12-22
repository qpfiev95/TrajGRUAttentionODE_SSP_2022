import torch
from torch import nn
from configuration.config import cfg
import torch.nn.functional as F
from model_components.base_conv_rnn import BaseConvRNN
from configuration.make_layers import make_convnet

# input: B, C, H, W
# flow: [B, 2, H, W]
def wrap(input, flow):
    B, C, H, W = input.size()
    # mesh grid
    # xx, yy: tensor (64,64)
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(input.device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(input.device)
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
    output = torch.nn.functional.grid_sample(input=input, grid=vgrid, mode='bilinear', align_corners=False)
    return output

class TrajGRU_Attention_ODE_EN(BaseConvRNN):
    ### This class is used for computing one timestep
    def __init__(self, input_channel, num_filter, b_h_w, zoneout=0.0,
                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                 h2h_kernel=(5, 5), h2h_stride=(1, 1), h2h_dilate=(1, 1), h2h_pad=(1,1), L=5,
                 act_type=cfg.MODEL.RNN_ACT_TYPE, first_layer=True, last_layer=False,
                 self_attention=True, ode_flag=False, zz_connection=False, device=None):
        super(TrajGRU_Attention_ODE_EN, self).__init__(num_filter=num_filter,
                                               b_h_w=b_h_w,
                                               h2h_kernel=h2h_kernel,
                                               h2h_dilate=h2h_dilate,
                                               i2h_kernel=i2h_kernel,
                                               i2h_pad=i2h_pad,
                                               i2h_stride=i2h_stride,
                                               act_type=act_type,
                                               prefix='TrajGRU_Attention_ODE_EN')
        self._L = L
        self.device =device
        self.zz_connection = zz_connection
        self.ode_flag = ode_flag
        self.self_attention = self_attention
        self._zoneout = zoneout
        self.first_layer = first_layer
        self.last_layer = last_layer
        # Trainable parameters:
        self.i2f_conv1 = nn.Conv2d(in_channels=input_channel,
                                   out_channels=num_filter,  # 32
                                   kernel_size=i2h_kernel,
                                   stride=i2h_stride,
                                   padding=i2h_pad)
        self.h2f_conv1 = nn.Conv2d(in_channels=input_channel,
                                   out_channels=num_filter,  # 32
                                   kernel_size=h2h_kernel,
                                   stride=h2h_stride,
                                   padding=h2h_pad)
        self.flows_conv = nn.Conv2d(in_channels=num_filter,  # 32
                                    out_channels=self._L * 2,
                                    kernel_size=h2h_kernel,
                                    stride=h2h_stride,
                                    padding=h2h_pad)
        self.ret = nn.Conv2d(in_channels=self._num_filter * self._L,
                             out_channels=self._num_filter * 3,
                             kernel_size=(1, 1),
                             stride=1)
        self.w_x = nn.Conv2d(in_channels=input_channel,
                             out_channels=num_filter * 3,
                             kernel_size=i2h_kernel,
                             stride=i2h_stride,
                             padding=i2h_pad)
        if self.self_attention:
            self.w_a_1 = nn.Conv2d(in_channels=num_filter,
                                   out_channels=num_filter,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            self.w_a_2 = nn.Conv2d(in_channels=num_filter,
                                   out_channels=num_filter,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            self.w_a_combine = nn.Conv2d(in_channels=num_filter * 2,
                                         out_channels=num_filter,
                                         kernel_size=1,
                                         stride=1)
            # self.w_avg = nn.AvgPool2d((b_h_w[1], b_h_w[2]),
            #                          stride=1)
            self.w_alignment_scores = nn.Linear(in_features=num_filter * 4,
                                                out_features=num_filter)

        if self.ode_flag:
            self.w_ode = nn.Conv2d(in_channels=input_channel,
                                 out_channels=num_filter * 3,
                                 kernel_size=i2h_kernel,
                                 stride=i2h_stride,
                                 padding=i2h_pad)
            #self.ode_net = make_convnet(n_inputs=num_filter, n_outputs=num_filter,
            #                   n_layers=1, n_units=48)

        if self.zz_connection:
            self.zz_i = nn.Conv2d(in_channels=input_channel,
                                 out_channels=num_filter*2,
                                 kernel_size=i2h_kernel,
                                 stride=i2h_stride,
                                 padding=i2h_pad)
            self.zz_h = nn.Conv2d(in_channels=input_channel,
                                 out_channels=num_filter*2,
                                 kernel_size=i2h_kernel,
                                 stride=i2h_stride,
                                 padding=i2h_pad)

    def motion_characteristics_generator(self, inputs, timestep):
        '''
        :param inputs: SxBxCxHxW
        :param timestep: Sx1
        :return:
        [time, velocity, acceleration]: SxBx3
        '''

        S, B, C, H, W = inputs.size()
        duration = timestep[1:] - timestep[:(S-1)]
        duration = torch.cat([timestep[0].unsqueeze(0), duration], dim=0)
        duration = duration.unsqueeze(1).unsqueeze(2).repeat(1, B, C) # SxBxC
        vel = torch.abs(inputs[1:] - inputs[:(S-1)])
        vel = torch.cat([inputs[0].unsqueeze(0), vel], dim=0) #SxBxCxHxW
        ###
        vel = torch.mean(vel, dim=(3,4)) #SxBxC
        vel /= duration #SxBxC
        acc = vel[1:] - vel[:(S-1)]
        acc = torch.cat([vel[0].unsqueeze(0), acc], dim=0)
        acc /= duration
        time = timestep.unsqueeze(1).unsqueeze(2).repeat(1, B, C)
        motion = torch.cat([time, vel, acc], dim=2) # SxBx3C
        motion = torch.reshape(motion, (-1, C*3)) # SBx3C
        return motion

    def attention_generator(self, states, state, timestep):
        '''
        :param: states: SxBxCxHxW
        :param state: BxCxHxW
        :return:
        a_state: BxCxHxW
        '''
        S, B, C, H, W = states.size()
        input_1 = states  # S_t x B x C x H x W
        input_2 = state  # B x C x H x W
        input_2_reshape = input_2.view(1, B, C, H, W).repeat(S, 1, 1, 1, 1)
        a_w_1 = self.w_a_1(torch.reshape(
            input_1 - input_2_reshape, (-1, C, H, W)))  # (S_t)*B x C x H x W
        a_w_2 = self.w_a_2(input_2)  # B x C x H x W
        a_w_2 = a_w_2.view(1, B, C, H, W).repeat(S, 1, 1, 1, 1)  # (S_t) x B x C x H x W
        a_w = torch.tanh(a_w_1 + torch.reshape(a_w_2, (-1, C, H, W)))  # (S_t)*B x C x H x W
        ###
        a_w = torch.reshape(a_w, (-1, B, C, H, W))  # (S_t) x B x C x H x W
        a_w = torch.mean(a_w, dim=(3, 4))  # SxBxC
        ###
        # a_w = self.w_avg(a_w) # SB x C x 1 x 1
        # a_w = torch.reshape(a_w, (S, B, C)) # SxBxC
        if S != 1:
            a_w = torch.reshape(a_w, (-1, C))  # SBxC
            motions = self.motion_characteristics_generator(states, timestep)
            alignment_score = torch.cat([motions.float(), a_w], dim=1)
            # alignment_score = self.w_alignment_scores(motions.float()) # SBxC
            alignment_score = self.w_alignment_scores(alignment_score)  # SBxC
            a_w_final = (torch.reshape(alignment_score, (S, B, C)).unsqueeze(3).unsqueeze(4))
            a_w_final = torch.softmax(a_w_final, dim=0)  # (S)xBxCx1x1
        else:
            a_w_final = a_w.unsqueeze(3).unsqueeze(4)
            a_w_final = torch.softmax(a_w_final, dim=0)  # (S)xBxCx1x1
        ####
        a_state = states * a_w_final  # SxBxCxHxW
        a_state = torch.sum(a_state, dim=0)
        a_state_final = self.w_a_combine(torch.cat([a_state, state], dim=1))
        return a_state_final

    def _flow_generator(self, inputs, states):
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)
        else:
            i2f_conv1 = None
        h2f_conv1 = self.h2f_conv1(states)
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self._act_type(f_conv1)  # leaky relu
        flows = self.flows_conv(f_conv1)  # B x 2L x H x W
        flows = torch.split(flows, 2, dim=1)  # list: L: B x 2 x H x W
        return flows

    def combined_state_generator(self, state, a_state):
        a_state_conv = self.zz_h(a_state)
        a_state_conv_slice = torch.split(a_state_conv, self._num_filter, dim=1)
        state_conv = self.zz_i(state)
        state_conv_slice = torch.split(state_conv, self._num_filter, dim=1)
        update_gate = torch.sigmoid(a_state_conv_slice[0] + state_conv_slice[0])
        update_gate_a = torch.sigmoid(a_state_conv_slice[1] + state_conv_slice[1])
        combined_state = update_gate * state + update_gate_a * a_state
        return combined_state

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
        elif zz_state is not None:
            state = self.combined_state_generator(states[-1], zz_state)
        else:
            state = states[-1]
        ## 1) Input convolution
        if inputs is not None:
            if self.self_attention:
                recent_input = self.attention_generator(inputs, inputs[-1], timestep[:(timestep_index+1)])
                input_conv = self.w_x(recent_input)
            else:
                recent_input = inputs[-1]
                input_conv = self.w_x(recent_input)  # Bx6CxHxW
            input_conv_slice = torch.split(input_conv, self._num_filter, dim=1)  # 6[BxCxHxW]
        ## 2) State convolution
        # Flow generator
        flows = self._flow_generator(recent_input, state)
        wrapped_data = []
        for j in range(len(flows)):
            flow = flows[j]
            wrapped_data.append(wrap(state, -flow))
        ## 2) State convolution
        wrapped_data = torch.cat(wrapped_data, dim=1)
        state_conv = self.ret(wrapped_data)  # Bx3CxHxW
        state_conv_slice = torch.split(state_conv, self._num_filter, dim=1)
        ## 3) ConvGRU operations
        if self.ode_flag:
            if timestep_index == len(timestep_index) - 1:
                delta = torch.Tensor([1]).to(cfg.GLOBAL.DEVICE)
            else:
                delta = timestep[timestep_index+1] - timestep[timestep_index]
            delta_ode_state = self.ode_net(state) * delta
            ode_state_conv = self.w_ode(delta_ode_state + state)
            ode_state_conv_slice = torch.split(ode_state_conv, self._num_filter, dim=1)
            if self.self_attention:
                reset_gate = torch.sigmoid(input_conv_slice[0] + state_conv_slice[0]
                                           + ode_state_conv_slice[0])
                update_gate = torch.sigmoid(input_conv_slice[1] + state_conv_slice[1]
                                            + ode_state_conv_slice[1])
                new_mem = torch.tanh(input_conv_slice[2] + reset_gate * state_conv_slice[2]
                                     + ode_state_conv_slice[2])
                new_state = update_gate * state + (1 - update_gate) * new_mem
                if states is not None:
                    new_a_state = self.attention_generator(states, new_state)
                else:
                    new_a_state = new_state
            else:
                reset_gate = torch.sigmoid(input_conv_slice[0] + state_conv_slice[0]
                                           + ode_state_conv_slice[0])
                update_gate = torch.sigmoid(input_conv_slice[1] + state_conv_slice[1]
                                            + ode_state_conv_slice[1])
                new_mem = torch.tanh(input_conv_slice[2] + reset_gate * state_conv_slice[2]
                                     + ode_state_conv_slice[2])
                new_state = update_gate * state + (1 - update_gate) * new_mem
                new_a_state = new_state
        else:
            if self.self_attention:
                reset_gate = torch.sigmoid(input_conv_slice[0] + state_conv_slice[0])
                update_gate = torch.sigmoid(input_conv_slice[1] + state_conv_slice[1])
                new_mem = torch.tanh(input_conv_slice[2] + reset_gate * state_conv_slice[2])
                new_state = update_gate * state + (1 - update_gate) * new_mem
                if states is not None:
                    new_a_state = new_state
                else:
                    new_a_state = new_state
            else:
                reset_gate = torch.sigmoid(input_conv_slice[0] + state_conv_slice[0])
                update_gate = torch.sigmoid(input_conv_slice[1] + state_conv_slice[1])
                new_mem = torch.tanh(input_conv_slice[2] + reset_gate * state_conv_slice[2])
                new_state = update_gate * state + (1 - update_gate) * new_mem
                new_a_state = new_state
        ###
        return new_state, new_a_state

class TrajGRU_Attention_ODE_DE(BaseConvRNN):
    ### This class is used for computing one timestep
    def __init__(self, input_channel, num_filter, b_h_w, zoneout=0.0,
                 i2h_kernel=3, i2h_stride=1, i2h_pad=1,
                 h2h_kernel=5, h2h_stride=1, h2h_pad=1, h2h_dilate=1, L=5,
                 act_type=cfg.MODEL.RNN_ACT_TYPE, first_layer=True, last_layer=False,
                 attention=True, self_attention=False, ode_flag=False, zz_connection=False, device=None):
        super(TrajGRU_Attention_ODE_DE, self).__init__(num_filter=num_filter,
                                               b_h_w=b_h_w,
                                               h2h_kernel=h2h_kernel,
                                               h2h_dilate=h2h_dilate,
                                               i2h_kernel=i2h_kernel,
                                               i2h_pad=i2h_pad,
                                               i2h_stride=i2h_stride,
                                               act_type=act_type,
                                               prefix='TrajGRU_Attention_ODE_DE')
        self._L = L
        self.device = device
        self.zz_connection = zz_connection
        self.ode_flag = ode_flag
        self.attention = attention
        self.self_attention = self_attention
        self._zoneout = zoneout
        self.first_layer = first_layer
        self.last_layer = last_layer
        # Trainable parameters:
        self.i2f_conv1 = nn.Conv2d(in_channels=input_channel,
                                   out_channels=num_filter,  # 32
                                   kernel_size=i2h_kernel,
                                   stride=i2h_stride,
                                   padding=i2h_pad)
        self.h2f_conv1 = nn.Conv2d(in_channels=input_channel,
                                   out_channels=num_filter,  # 32
                                   kernel_size=h2h_kernel,
                                   stride=h2h_stride,
                                   padding=h2h_pad)
        self.flows_conv = nn.Conv2d(in_channels=num_filter,  # 32
                                    out_channels=self._L * 2,
                                    kernel_size=h2h_kernel,
                                    stride=h2h_stride,
                                    padding=h2h_pad)
        self.ret = nn.Conv2d(in_channels=self._num_filter * self._L,
                             out_channels=self._num_filter * 3,
                             kernel_size=(1, 1),
                             stride=1)
        self.w_x = nn.Conv2d(in_channels=input_channel,
                             out_channels=num_filter * 3,
                             kernel_size=i2h_kernel,
                             stride=i2h_stride,
                             padding=i2h_pad)
        if self.attention:
            self.w_a_1 = nn.Conv2d(in_channels=num_filter,
                                   out_channels=num_filter,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            self.w_a_2 = nn.Conv2d(in_channels=num_filter,
                                   out_channels=num_filter,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            #self.w_a_combine = nn.Conv2d(in_channels=num_filter * 2,
            #                       out_channels=num_filter,
            #                       kernel_size=1,
            #                       stride=1)
            #self.w_avg = nn.AvgPool2d((b_h_w[1], b_h_w[2]),
            #                          stride=1)
            self.w_alignment_scores = nn.Linear(in_features=num_filter*4,
                                                out_features=num_filter)
            self.w_a_h = nn.Conv2d(in_channels=num_filter,
                                   out_channels=num_filter*3,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            self.w_a = nn.Conv2d(in_channels=num_filter,
                                   out_channels=num_filter*3,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        ###################################
        if self.ode_flag:
            self.w_ode = nn.Conv2d(in_channels=input_channel,
                                   out_channels=num_filter * 3,
                                   kernel_size=i2h_kernel,
                                   stride=i2h_stride,
                                   padding=i2h_pad)
            self.w_ode_combine = nn.Conv2d(in_channels=input_channel * 2,
                                   out_channels=num_filter,
                                   kernel_size=i2h_kernel,
                                   stride=i2h_stride,
                                   padding=i2h_pad)
            self.w_new_h = nn.Conv2d(in_channels=input_channel,
                                   out_channels=num_filter * 3,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            if self.zz_connection:
                self.zz_ode = nn.Conv2d(in_channels=input_channel,
                                           out_channels=num_filter,  # 32
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)

        if self.zz_connection:
            self.zz_i = nn.Conv2d(in_channels=input_channel,
                                 out_channels=num_filter,
                                 kernel_size=i2h_kernel,
                                 stride=i2h_stride,
                                 padding=i2h_pad)
            self.zz_h = nn.Conv2d(in_channels=input_channel,
                                 out_channels=num_filter,
                                 kernel_size=h2h_kernel,
                                 stride=h2h_stride,
                                 padding=h2h_pad)

        #if self.zz_connection:
        #    self.zz_i = nn.Conv2d(in_channels=input_channel,
        #                         out_channels=num_filter * 2,
        #                         kernel_size=i2h_kernel,
        #                         stride=i2h_stride,
        #                        padding=i2h_pad)
        #    self.zz_h = nn.Conv2d(in_channels=input_channel,
        #                         out_channels=num_filter * 2,
        #                        kernel_size=h2h_kernel,
        #                        stride=h2h_stride,
        #                        padding=h2h_pad)

    def combined_state_generator(self, state, zz_state, ode_state):
        if ode_state is not None:
            update_gate = torch.sigmoid(self.zz_i(state) + self.zz_h(zz_state) + self.zz_ode(ode_state))
        else:
            update_gate = torch.sigmoid(self.zz_i(state) + self.zz_h(zz_state))
        combined_state = state + update_gate*zz_state
        return combined_state

    #def combined_state_generator(self, state, a_state):
    #    a_state_conv = self.zz_h(a_state)
    #    a_state_conv_slice = torch.split(a_state_conv, self._num_filter, dim=1)
    #    state_conv = self.zz_i(state)
    #    state_conv_slice = torch.split(state_conv, self._num_filter, dim=1)
    #    update_gate = torch.sigmoid(a_state_conv_slice[0] + state_conv_slice[0])
    #    update_gate_a = torch.sigmoid(a_state_conv_slice[1] + state_conv_slice[1])
    #    combined_state = update_gate * state + update_gate_a * a_state
    #    return combined_state

    def _flow_generator(self, inputs, states):
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)
        else:
            i2f_conv1 = None
        h2f_conv1 = self.h2f_conv1(states)
        ### Notice the size of the tensors
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self._act_type(f_conv1)  # leaky relu
        flows = self.flows_conv(f_conv1)  # B x 2L x H x W
        flows = torch.split(flows, 2, dim=1)  # list: L: B x 2 x H x W
        return flows

    def motion_characteristics_generator(self, inputs, timestep):
        '''
        :param inputs: SxBxCxHxW
        :param timestep: Sx1
        :return:
        [time, velocity, acceleration]: SxBx3
        '''

        S, B, C, H, W = inputs.size()
        duration = timestep[1:] - timestep[:(S-1)]
        duration = torch.cat([timestep[0].unsqueeze(0), duration], dim=0)
        duration = duration.unsqueeze(1).unsqueeze(2).repeat(1, B, C) # SxBxC
        vel = inputs[1:] - inputs[:(S-1)]
        vel = torch.cat([inputs[0].unsqueeze(0), vel], dim=0) #SxBxCxHxW
        ###
        vel = torch.mean(vel, dim=(3,4)) #SxBxC
        vel /= duration #SxBxC
        acc = vel[1:] - vel[:(S-1)]
        acc = torch.cat([vel[0].unsqueeze(0), acc], dim=0)
        acc /= duration
        time = timestep.unsqueeze(1).unsqueeze(2).repeat(1, B, C)
        motion = torch.cat([time, vel, acc], dim=2) # SxBx3C
        motion = torch.reshape(motion, (-1, C*3)) # SBx3C
        return motion

    def attention_generator(self, states, state, timestep):
        '''
        :param: states: SxBxCxHxW
        :param state: BxCxHxW
        :return:
        a_state: BxCxHxW
        '''
        S, B, C, H, W = states.size()
        input_1 = states  # S_t x B x C x H x W
        input_2 = state # B x C x H x W
        #input_2_reshape = input_2.view(1, B, C, H, W).repeat(S, 1, 1, 1, 1)
        a_w_1 = self.w_a_1(torch.reshape(input_1, (-1, C, H, W)))  # (S_t)*B x C x H x W
        a_w_2 = self.w_a_2(input_2)  # B x C x H x W
        a_w_2 = a_w_2.view(1, B, C, H, W).repeat(S, 1, 1, 1, 1)  # (S_t) x B x C x H x W
        a_w = torch.tanh(a_w_1 + torch.reshape(a_w_2, (-1, C, H, W)))  # (S_t)*B x C x H x W
        ###
        a_w = torch.reshape(a_w, (-1, B, C, H, W))  # (S_t) x B x C x H x W
        a_w = torch.mean(a_w, dim=(3,4)) # SxBxC
        ###
        #a_w = self.w_avg(a_w) # SB x C x 1 x 1
        #a_w = torch.reshape(a_w, (S, B, C)) # SxBxC
        if S != 1:
            a_w = torch.reshape(a_w, (-1, C))  # SBxC
            motions = self.motion_characteristics_generator(states, timestep)
            alignment_score = torch.cat([motions.float(), a_w], dim=1)
            #alignment_score = self.w_alignment_scores(motions.float()) # SBxC
            alignment_score = self.w_alignment_scores(alignment_score)  # SBxC
            a_w_final = (torch.reshape(alignment_score,(S,B,C)).unsqueeze(3).unsqueeze(4))
            a_w_final = torch.softmax(a_w_final, dim=0)  # (S)xBxCx1x1
        else:
            a_w_final = a_w.unsqueeze(3).unsqueeze(4)
            a_w_final = torch.softmax(a_w_final, dim=0)  # (S)xBxCx1x1
        ####
        #a_state_final = self.w_a_combine(torch.cat([a_state, state], dim=1))
        a_state = states * a_w_final
        a_state = torch.sum(a_state, dim=0)
        ###
        h = self.w_a_h(state)
        h_slice = torch.split(h, self._num_filter, dim=1)
        a = self.w_a(a_state)
        a_slice = torch.split(a, self._num_filter, dim=1)
        ###
        update_gate_0 = torch.sigmoid(h_slice[0] + a_slice[0])
        update_gate_1 = torch.sigmoid(h_slice[1] + a_slice[1])
        new_mem = torch.tanh(h_slice[2] + a_slice[2])
        a_state_final = update_gate_0 * state + update_gate_1 * new_mem
        return a_state_final

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
                state = self.combined_state_generator(states[-1], zz_state, ode_states[timestep_index+1])
                # state = self.combined_state_generator(states[-1], zz_state)
            else:
                state = self.combined_state_generator(states[-1], zz_state, None)
                # state = self.combined_state_generator(states[-1], zz_state)
        else:
            state = states[-1]
        ## 1) Input convolution
        # B, C, H, W = input.size()
        if not self.zz_connection:
            if self.attention:
                recent_input = inputs[-1]
                input_conv = self.w_x(recent_input)
                input_conv_slice = torch.split(input_conv, self._num_filter, dim=1)  # 5[BxCxHxW]
                flows = self._flow_generator(recent_input, state)
            elif inputs is not None:
                recent_input = inputs[-1]
                input_conv = self.w_x(recent_input)  # Bx6CxHxW
                input_conv_slice = torch.split(input_conv, self._num_filter, dim=1)  # 5[BxCxHxW]
                flows = self._flow_generator(recent_input, state)
            else:
                recent_input = None
                #input_conv_slice = torch.split(input_conv, self._num_filter, dim=1)  # 5[BxCxHxW]
                flows = self._flow_generator(recent_input, state)
            #input_conv_slice = torch.split(input_conv, self._num_filter, dim=1)  # 5[BxCxHxW]
            #flows = self._flow_generator(recent_input, state)
        else:
            recent_input = None
            flows = self._flow_generator(recent_input, state)
        wrapped_data = []
        for j in range(len(flows)):
            flow = flows[j]
            wrapped_data.append(wrap(state, -flow))
        ## 2) State convolution
        wrapped_data = torch.cat(wrapped_data, dim=1)
        state_conv = self.ret(wrapped_data)  # Bx3CxHxW
        state_conv_slice = torch.split(state_conv, self._num_filter, dim=1)
        ## 3) ConvGRU operations
        if self.ode_flag:
            ode_conv_combine = self.w_ode_combine(torch.cat([ode_states[timestep_index], ode_states[timestep_index + 1]], dim=1))
            ode_conv = self.w_ode(ode_conv_combine)
            ode_conv_slice = torch.split(ode_conv, self._num_filter, dim=1)
            if self.attention:
                #state = self.attention_generator(states[-10:], states[-1],
                #    timestep[(timestep_index):(timestep_index) + 10])
                if recent_input is not None:
                    reset_gate = torch.sigmoid(input_conv_slice[0] + state_conv_slice[0])
                    update_gate = torch.sigmoid(input_conv_slice[1] + state_conv_slice[1])
                    new_mem = torch.tanh(input_conv_slice[2] + reset_gate * state_conv_slice[2])
                    new_state = update_gate * state + (1 - update_gate) * new_mem
                    ###
                    new_h_conv = self.w_new_h(new_state)
                    new_h_conv_slice = torch.split(new_h_conv, self._num_filter, dim=1)
                    update_gate_ode = torch.sigmoid(new_h_conv_slice[0] + ode_conv_slice[0])
                    reset_gate_ode = torch.sigmoid(new_h_conv_slice[1] + ode_conv_slice[1])
                    new_mem_ode = torch.tanh(new_h_conv_slice[1] + reset_gate_ode * ode_conv_slice[2])
                    new_state = update_gate_ode * new_state + (1 - update_gate_ode) * new_mem_ode
                    new_a_state = self.attention_generator(states[-10:], new_state,
                        timestep[(timestep_index):(timestep_index) + 10])
                    #new_a_state = new_state
                else:
                    reset_gate = torch.sigmoid(state_conv_slice[0])
                    update_gate = torch.sigmoid(state_conv_slice[1])
                    new_mem = torch.tanh(reset_gate * state_conv_slice[2])
                    new_state = update_gate * state + (1 - update_gate) * new_mem
                    ###
                    new_h_conv = self.w_new_h(new_state)
                    new_h_conv_slice = torch.split(new_h_conv, self._num_filter, dim=1)
                    update_gate_ode = torch.sigmoid(new_h_conv_slice[0] + ode_conv_slice[0])
                    reset_gate_ode = torch.sigmoid(new_h_conv_slice[1] + ode_conv_slice[1])
                    new_mem_ode = torch.tanh(new_h_conv_slice[1] + reset_gate_ode * ode_conv_slice[2])
                    new_state = update_gate_ode * new_state + (1 - update_gate_ode) * new_mem_ode
                    ###
                    new_a_state = self.attention_generator(states[-10:], new_state,
                                                           timestep[(timestep_index):(timestep_index) + 10])
                    #new_a_state = new_state
            else:
                if recent_input is not None:
                    reset_gate = torch.sigmoid(input_conv_slice[0] + state_conv_slice[0])
                    update_gate = torch.sigmoid(input_conv_slice[1] + state_conv_slice[1])
                    new_mem = torch.tanh(input_conv_slice[2] + reset_gate * state_conv_slice[2])
                    new_state = update_gate * state + (1 - update_gate) * new_mem
                    ###
                    new_h_conv = self.w_new_h(new_state)
                    new_h_conv_slice = torch.split(new_h_conv, self._num_filter, dim=1)
                    update_gate_ode = torch.sigmoid(new_h_conv_slice[0] + ode_conv_slice[0])
                    reset_gate_ode = torch.sigmoid(new_h_conv_slice[1] + ode_conv_slice[1])
                    new_mem_ode = torch.tanh(new_h_conv_slice[1] + reset_gate_ode * ode_conv_slice[2])
                    new_state = update_gate_ode * new_state + (1 - update_gate_ode) * new_mem_ode
                    ###
                    new_a_state = new_state
                else:
                    reset_gate = torch.sigmoid(state_conv_slice[0])
                    update_gate = torch.sigmoid(state_conv_slice[1])
                    new_mem = torch.tanh(reset_gate * state_conv_slice[2])
                    new_state = update_gate * state + (1 - update_gate) * new_mem
                    ###
                    new_h_conv = self.w_new_h(new_state)
                    new_h_conv_slice = torch.split(new_h_conv, self._num_filter, dim=1)
                    update_gate_ode = torch.sigmoid(new_h_conv_slice[0] + ode_conv_slice[0])
                    reset_gate_ode = torch.sigmoid(new_h_conv_slice[1] + ode_conv_slice[1])
                    new_mem_ode = torch.tanh(new_h_conv_slice[1] + reset_gate_ode * ode_conv_slice[2])
                    new_state = update_gate_ode * new_state + (1 - update_gate_ode) * new_mem_ode
                    ###
                    new_a_state = new_state
        else:
            if self.attention:
                if recent_input is not None:
                    reset_gate = torch.sigmoid(input_conv_slice[0] + state_conv_slice[0])
                    update_gate = torch.sigmoid(input_conv_slice[1] + state_conv_slice[1])
                    new_mem = torch.tanh(input_conv_slice[2]
                                         + reset_gate * state_conv_slice[2])
                    new_state = update_gate * state + (1 - update_gate) * new_mem
                    #new_a_state = self.attention_generator(states[-10:], new_state)
                    new_a_state = self.attention_generator(
                        torch.cat([states[-9:], new_state.unsqueeze(0)], dim=0), new_state,
                        timestep[(timestep_index + 1):(timestep_index + 1) + 10])
                    #new_a_state = new_state
                else:
                    reset_gate = torch.sigmoid(state_conv_slice[0])
                    update_gate = torch.sigmoid(state_conv_slice[1])
                    new_mem = torch.tanh(reset_gate * state_conv_slice[2])
                    new_state = update_gate * state + (1 - update_gate) * new_mem
                    new_a_state = self.attention_generator(
                        torch.cat([states[-9:], new_state.unsqueeze(0)], dim=0), new_state,
                        timestep[(timestep_index + 1):(timestep_index + 1) + 10])
                    #new_a_state = new_state
            else:
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

class TrajGRU_ODE(BaseConvRNN):
    ### This class is used for computing one timestep
    def __init__(self, input_channel, num_filter, b_h_w, zoneout=0.0,
                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                 h2h_kernel=(5, 5), h2h_dilate=(1, 1), L=5,
                 act_type=cfg.MODEL.RNN_ACT_TYPE, first_layer=True, last_layer=False,
                 attention=True, self_attention=False, ode_flag=False, zz_connection=False):
        super(TrajGRU_ODE, self).__init__(num_filter=num_filter,
                                               b_h_w=b_h_w,
                                               h2h_kernel=h2h_kernel,
                                               h2h_dilate=h2h_dilate,
                                               i2h_kernel=i2h_kernel,
                                               i2h_pad=i2h_pad,
                                               i2h_stride=i2h_stride,
                                               act_type=act_type,
                                               prefix='TrajGRU_A_ODE_ZZ_FO')
        self._L = L
        self.zz_connection = zz_connection
        self.ode_flag = ode_flag
        self.attention = attention
        self.self_attention = self_attention
        self._zoneout = zoneout
        self.first_layer = first_layer
        self.last_layer = last_layer
        # Trainable parameters:

    def forward(self, timestep, inputs, states, ode_states, zz_state):
        new_state = ode_states[timestep + 1]
        new_a_state = new_state
        return new_state, new_a_state
