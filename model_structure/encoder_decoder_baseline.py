from torch import nn
import torch
from configuration.make_layers import make_layers
from configuration.config import cfg

class Encoder(nn.Module):
    def __init__(self, subnets, rnns, time_step_orders):
        super().__init__()
        assert len(subnets)==len(rnns)
        self.time_step_orders = time_step_orders
        self.num_layer = len(subnets)
        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage'+str(index), make_layers(params))
            setattr(self, 'rnn'+str(index), rnn)
    def forward_by_timestep(self, time_index, input,
                            inputs_list, states_list, timestep):
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
                        time_index, inputs_list[i - 1], states_list[i - 1], None, None, timestep)
                elif i == self.num_layer:
                    new_state, new_a_state = getattr(self, 'rnn' + str(i))(
                        time_index, inputs_list[i - 1], states_list[i - 1], None, None, timestep)
                else:
                    new_state, new_a_state = getattr(self, 'rnn' + str(i))(
                        time_index, inputs_list[i - 1], states_list[i - 1], None, None, timestep)
                input = new_a_state
                states_list[i - 1] = torch.cat([states_list[i - 1], torch.unsqueeze(new_state, dim=0)], dim=0)
        return inputs_list, states_list

    # inputs: 5D S*B*C*H*W
    def forward(self, data_inputs, timestep):
        '''
        :param inputs: SxBxCxHxW
        :return:
        '''
        # At the first timestep: states = []
        inputs_list, states_list = \
            self.forward_by_timestep(0, data_inputs[0], [], [], timestep)
        for i in range(1, len(self.time_step_orders)):
            inputs_list, states_list = \
                self.forward_by_timestep(i, data_inputs[i], inputs_list, states_list, timestep)
        ###
        return inputs_list, states_list

class Decoder(nn.Module):
    def __init__(self, subnets, rnns, time_step_orders):
        super().__init__()
        self.time_step_orders = time_step_orders
        self.num_layer = len(rnns)
        for index, rnn in enumerate(rnns):
            setattr(self, 'rnn' + str(self.num_layer - index), rnn)
        for index, params in enumerate(subnets):
            setattr(self, 'stage' + str(self.num_layer + 1 - index), make_layers(params))

    def forward_by_timestep(self, time_index, inputs_list, states_list, total_timestep):
        '''
        :param time_index: int
        :param states: List of K hidden states
        :param zz_state: B x C x H x W
        :param inputs_top_layer: S_t x B x C x H x W
        :return:
        '''
        # At the top layer:
        new_state, new_a_state = getattr(self, 'rnn' + str(self.num_layer))(
                time_index, None, states_list[-1], None, None, total_timestep)
        states_list[-1] = torch.cat([states_list[-1], torch.unsqueeze(new_state, dim=0)], dim=0)
        input = getattr(self, 'stage' + str(self.num_layer+1))(new_a_state)

        for i in range(1, self.num_layer)[::-1]:  # 2 1
            inputs_list[i-1] = torch.cat([inputs_list[i-1], torch.unsqueeze(input, dim=0)], dim=0)
            new_state, new_a_state = getattr(self, 'rnn' + str(i))(
                time_index, inputs_list[i-1], states_list[i-1], None, None, total_timestep)
            states_list[i-1] = torch.cat([states_list[i-1], torch.unsqueeze(new_state, dim=0)], dim=0)
            if i == 1:
                input = getattr(self, 'stage' + str(i + 1))(new_state)
            else:
                input = getattr(self, 'stage' + str(i + 1))(new_a_state)
        output = getattr(self, 'stage' + str(1))(input)
        return states_list, output

    def forward(self, inputs_list, states_list, timestep_de, timestep_en):
        '''
        :param states_list: list of K-1 tensor 1xBxCxHxW
        :param states_list: list of K tensor 1xBxCxHxW
        :return:
        '''
        outputs = []

        total_timestep = torch.cat([timestep_en, timestep_de])
        for i in range(0, len(self.time_step_orders)):
            states_list, output = \
                self.forward_by_timestep(
                    i, inputs_list, states_list, total_timestep)
            outputs.append(output)
        return torch.stack(outputs, dim=0)

class ED(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, input, timestep_en, timestep_de):
        inputs_list, states_list = self.encoder(input, timestep_en)
        output = self.decoder(inputs_list, states_list, timestep_de, timestep_en)
        return output

