from torch import nn
import torch
from configuration.make_layers import make_layers
from configuration.config import cfg

class Encoder(nn.Module):
    def __init__(self, subnets, rnns, subnet_zigzag, time_step_orders):
        super().__init__()
        assert len(subnets)==len(rnns)
        self.time_step_orders = time_step_orders
        self.num_layer = len(subnets)
        ###
        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage'+str(index), make_layers(params))
            setattr(self, 'rnn'+str(index), rnn)
        ### Using to connect the previpus last layer and the next first layer
        self.subnet_deconv_zz = make_layers(subnet_zigzag[0])
        self.subnet_conv_zz = make_layers(subnet_zigzag[1])
    def forward_by_timestep(self, time_index, input,
                            inputs_list, states_list, zz_state, timestep):
        '''
        :param time_step: int
        :param inputs: St x B x C x H x W
        :param states_list: List of K hidden states SxBxCxHxW
        :return:
        '''
        for i in range(1, self.num_layer+1):
            input = getattr(self, 'stage' + str(i))(input)
            if time_index == 0: # At the first timestep
                inputs_list.append(torch.unsqueeze(input, dim=0))
                if i == self.num_layer:
                    new_state, new_a_state = getattr(self, 'rnn' + str(i))(
                        time_index, inputs_list[i - 1], None, None, None, timestep)
                    zz_state = self.subnet_deconv_zz(new_state)
                elif i == 1:
                    new_state, new_a_state = getattr(self, 'rnn' + str(i))(
                        time_index, inputs_list[i - 1], None, None, None, timestep)
                else:
                    new_state, new_a_state = getattr(self, 'rnn' + str(i))(
                        time_index, inputs_list[i - 1], None, None, None, timestep)
                input = new_a_state
                states_list.append(torch.unsqueeze(new_state, dim=0))
            else:
                inputs_list[i-1] = torch.cat([inputs_list[i-1], torch.unsqueeze(input, dim=0)], dim=0)
                if i == 1:
                    new_state, new_a_state = getattr(self, 'rnn' + str(i))(
                        time_index, inputs_list[i - 1], states_list[i - 1], None, zz_state, timestep)
                    if time_index == len(self.time_step_orders) - 1:
                        zz_state = self.subnet_conv_zz(new_a_state)
                elif i == self.num_layer:
                    new_state, new_a_state = getattr(self, 'rnn' + str(i))(
                        time_index, inputs_list[i - 1], states_list[i - 1], None, None, timestep)
                    if time_index != len(self.time_step_orders)-1:
                        zz_state = self.subnet_deconv_zz(new_state)
                else:
                    new_state, new_a_state = getattr(self, 'rnn' + str(i))(
                        time_index, inputs_list[i - 1], states_list[i - 1], None, None, timestep)
                input = new_a_state
                states_list[i - 1] = torch.cat([states_list[i - 1], torch.unsqueeze(new_state, dim=0)], dim=0)
        return inputs_list, states_list, zz_state
    # inputs: 5D S*B*C*H*W
    def forward(self, data_inputs, timestep):
        '''
        :param inputs: SxBxCxHxW
        :return:
        '''
        # At the first timestep: states = []
        inputs_list, states_list, zz_state = \
            self.forward_by_timestep(0, data_inputs[0], [], [], None, timestep)
        for i in range(1, len(self.time_step_orders)):
            inputs_list, states_list, zz_state = \
                self.forward_by_timestep(i, data_inputs[i], inputs_list, states_list, zz_state, timestep)
        return inputs_list, states_list, zz_state

class Decoder(nn.Module):
    def __init__(self, subnets, rnns, ode_solvers, subnet_zigzag, time_step_orders, ode_flag):
        super().__init__()
        self.ode_flag = ode_flag
        self.time_step_orders = time_step_orders
        self.num_layer = len(rnns)
        for index, rnn in enumerate(rnns):
            setattr(self, 'rnn' + str(self.num_layer - index), rnn)
        for index, params in enumerate(subnets):
            setattr(self, 'stage' + str(self.num_layer + 1 - index), make_layers(params))
        if self.ode_flag:
            for index, (ode_solver) in enumerate(ode_solvers):
                setattr(self, 'odesolver' + str(self.num_layer - index), ode_solver)
        ### Using to connect the previpus last layer and the next first layer
        self.subnet_conv_zz_bottom = make_layers(subnet_zigzag[0])

    def forward_by_timestep(self, time_index, ode_states_list, inputs_list, states_list, zz_state, total_timestep):
        '''
        :param time_index: int
        :param states: List of K hidden states
        :param zz_state: B x C x H x W
        :param inputs_top_layer: S_t x B x C x H x W
        :return:
        '''
        # At the top layer:
        new_state, new_a_state = getattr(self, 'rnn' + str(self.num_layer))(
                time_index, None, states_list[-1], ode_states_list[-1], zz_state, total_timestep)
        states_list[-1] = torch.cat([states_list[-1], torch.unsqueeze(new_state, dim=0)], dim=0)
        input = getattr(self, 'stage' + str(self.num_layer+1))(new_a_state)

        for i in range(1, self.num_layer)[::-1]:  # 2 1
            inputs_list[i-1] = torch.cat([inputs_list[i-1], torch.unsqueeze(input, dim=0)], dim=0)
            new_state, new_a_state = getattr(self, 'rnn' + str(i))(
                time_index, inputs_list[i-1], states_list[i-1], ode_states_list[i-1], None, total_timestep)
            states_list[i-1] = torch.cat([states_list[i-1], torch.unsqueeze(new_state, dim=0)], dim=0)
            if i == 1:
                input = getattr(self, 'stage' + str(i + 1))(new_state)
                zz_state = self.subnet_conv_zz_bottom(
                    getattr(self, 'stage' + str(i + 1))(new_a_state))
            else:
                input = getattr(self, 'stage' + str(i + 1))(new_a_state)
        output = getattr(self, 'stage' + str(1))(input)
        return states_list, output, zz_state

    def forward(self, inputs_list, states_list, zz_state, timestep_de, timestep_en):
        '''
        :param states_list: list of K-1 tensor 1xBxCxHxW
        :param states_list: list of K tensor 1xBxCxHxW
        :return:
        '''
        outputs = []
        ode_states_list = []
        init_time = self.time_step_orders[0] - torch.Tensor([1]).to(cfg.GLOBAL.DEVICE)
        ode_time_steps = torch.cat([init_time, timestep_de])
        total_timestep = torch.cat([timestep_en, timestep_de])
        for j in range(1, self.num_layer+1): # 1 2 3

            if self.ode_flag:
                init_ode_state = states_list[j - 1][-1]
                ode_states = getattr(self, 'odesolver' + str(j))(ode_time_steps, init_ode_state)
                ode_states_list.append(ode_states)
            else:
                ode_states_list.append(None)

        for i in range(0, len(self.time_step_orders)):
            states_list, output, zz_state = \
                self.forward_by_timestep(
                    i, ode_states_list, inputs_list, states_list, zz_state, total_timestep)
            outputs.append(output)
        return torch.stack(outputs, dim=0)

class ED(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, timestep_en, timestep_de):
        inputs_list, states_list, zz_state = self.encoder(input, timestep_en)
        output = self.decoder(inputs_list, states_list, zz_state, timestep_de, timestep_en)
        return output

