import torch
import torch.nn as nn
import sys
from utils import get_device, get_norm_layer
from configuration.config import cfg
from configuration.make_layers import make_convnet
from ode_func import DiffeqSolver, ODEFunc
import numpy as np
#from helpers.model_summary import count_parameters

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_channel, hidden_channel, kernel_size, bias, dtype):
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_channel = hidden_channel
        self.bias = bias
        self.dtype = dtype

        self.i2h = nn.Conv2d(in_channels=input_channel,
                                    out_channels=3 * self.hidden_channel,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.h2h = nn.Conv2d(in_channels=hidden_channel,
                                    out_channels=3 * self.hidden_channel,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)
    # H0: B x C_hidden x H x W
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_channel, self.height, self.width)).type(self.dtype)

    def forward(self, input_tensor, h_cur, mask=None):
        """
        :param self:
        :param input_tensor: (b, c, h, w) / input is actually the target_model
        :param h_cur: (b, c_hidden, h, w) / current hidden and cell states respectively
        :return: h_next, next hidden state
        """
        i2h = self.i2h(input_tensor)
        i2h_slice = torch.split(i2h, self.hidden_channel, dim=1)
        h2h = self.h2h(h_cur)
        h2h_slice = torch.split(h2h, self.hidden_channel, dim=1)

        reset_gate = torch.sigmoid(i2h_slice[0] + h2h_slice[0])
        update_gate = torch.sigmoid(i2h_slice[1] + h2h_slice[1])
        new_mem = torch.tanh(i2h_slice[2] + reset_gate * h2h_slice[2])

        h_next = (1 - update_gate) * h_cur + update_gate * new_mem
        if mask is not None:
            mask = mask.view(-1, 1, 1, 1).expand_as(h_cur)
            h_next = mask * h_next + (1 - mask) * h_cur

        return h_next

class Encoder_z0_ODE_ConvGRU(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, dtype, batch_first=False,
                 bias=True, return_all_layers=False, z0_diffeq_solver=None, run_backwards=None):

        super(Encoder_z0_ODE_ConvGRU, self).__init__()
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.z0_diffeq_solver = z0_diffeq_solver
        self.run_backwards = run_backwards
        ##### By product for visualization
        self.by_product = {}
        ###
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_channel=cur_input_dim,
                                         hidden_channel=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

        # last conv layer for generating mu, sigma
        self.z0_dim = hidden_dim[0]
        z = hidden_dim[0]
        self.transform_z0 = nn.Sequential(
            nn.Conv2d(z, z, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(z, z * 2, 1, 1, 0), )

    def forward(self, input_tensor, time_steps, mask=None, tracker=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        assert (input_tensor.size(1) == len(time_steps)), "Sequence length should be same as time_steps"
        ###
        last_yi, latent_ys = self.run_ode_conv_gru(
            input_tensor=input_tensor,
            mask=mask,
            time_steps=time_steps,
            run_backwards=self.run_backwards,
            tracker=tracker)

        #last_yi, latent_ys = self.run_ode_conv_gru_forward(
        #    input_tensor=input_tensor,
        #    mask=mask,
        #    time_steps=time_steps,
        #    tracker=tracker)

        trans_last_yi = self.transform_z0(last_yi)

        mean_z0, std_z0 = torch.split(trans_last_yi, self.z0_dim, dim=1)
        std_z0 = std_z0.abs()

        return mean_z0, std_z0

    def run_ode_conv_gru(self, input_tensor, mask, time_steps, run_backwards=True, tracker=None):
        ### Input_shape: B x S x C x H x W
        b, t, c, h, w = input_tensor.size()
        device = get_device(input_tensor)
        # Set initial inputs
        prev_input_tensor = torch.zeros((b, c, h, w)).to(device)

        # Time configuration
        # Run ODE backwards and combine the y(t) estimates using gating
        prev_t, t_i = time_steps[-1] + 0.01, time_steps[-1]
        latent_ys = []

        time_points_iter = range(0, time_steps.size(-1))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        for idx, i in enumerate(time_points_iter):
            # return grad
            inc = self.z0_diffeq_solver.ode_func(prev_t, prev_input_tensor) * (t_i - prev_t)
            assert (not torch.isnan(inc).any())
            # init: prev_input = 0 then prev_input = yi
            ode_sol = prev_input_tensor + inc
            ode_sol = torch.stack((prev_input_tensor, ode_sol), dim=1)  # [1, b, 2, c, h, w] => [b, 2, c, h, w]
            assert (not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, 0, :] - prev_input_tensor) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :] - prev_input_tensor))
                exit()

            yi_ode = ode_sol[:, -1, :]
            xi = input_tensor[:, i, :]

            # only 1 now
            ### modified
            if mask is not None:
                yi = self.cell_list[0](input_tensor=xi,
                                       h_cur=yi_ode,
                                       mask=mask[:, i])
            else:
                yi = self.cell_list[0](input_tensor=xi,
                                       h_cur=yi_ode,
                                       mask=None)

            # return to iteration
            prev_input_tensor = yi
            prev_t, t_i = time_steps[i], time_steps[i - 1]
            latent_ys.append(yi)

        latent_ys = torch.stack(latent_ys, 1)
        return yi, latent_ys

    def run_ode_conv_gru_forward(self, input_tensor, mask, time_steps, tracker=None):
        ### Input_shape: B x S x C x H x W
        b, t, c, h, w = input_tensor.size()
        device = get_device(input_tensor)
        # Set initial inputs
        prev_input_tensor = torch.zeros((b, c, h, w)).to(device)

        # Time configuration
        # Run ODE backwards and combine the y(t) estimates using gating
        prev_t, t_i = time_steps[0] - 0.01, time_steps[0]
        latent_ys = []

        time_points_iter = range(0, time_steps.size(-1))

        for idx, i in enumerate(time_points_iter):
            # return grad
            inc = self.z0_diffeq_solver.ode_func(prev_t, prev_input_tensor) * (t_i - prev_t)
            assert (not torch.isnan(inc).any())
            # init: prev_input = 0 then prev_input = yi
            ode_sol = prev_input_tensor + inc
            ode_sol = torch.stack((prev_input_tensor, ode_sol), dim=1)  # [1, b, 2, c, h, w] => [b, 2, c, h, w]
            assert (not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, 0, :] - prev_input_tensor) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :] - prev_input_tensor))
                exit()

            yi_ode = ode_sol[:, -1, :]
            xi = input_tensor[:, i, :]

            # only 1 now
            ### modified
            yi = self.cell_list[0](input_tensor=xi,
                                   h_cur=yi_ode,
                                   mask=mask[:, i])

            # return to iteration
            prev_input_tensor = yi
            prev_t, t_i = time_steps[i], time_steps[i] + 0.05
            latent_ys.append(yi)

        latent_ys = torch.stack(latent_ys, 1)
        return yi, latent_ys

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
####################### Encoder-Forecaster ###############################################
class Encoder(nn.Module):
    def __init__(self, input_dim=1, ch=64, n_downs=2):
        super(Encoder, self).__init__()
        model = []
        model += [nn.Conv2d(input_dim, ch, 3, 1, 1)]
        model += [get_norm_layer(ch)]
        model += [nn.ReLU()]
        for _ in range(n_downs):
            model += [nn.Conv2d(ch, ch * 2, 4, 2, 1)]
            model += [get_norm_layer(ch * 2)]
            model += [nn.ReLU()]
            ch *= 2
        self.model = nn.Sequential(*model)
    def forward(self, x):
        out = self.model(x)
        return out

class Forecaster(nn.Module):
    def __init__(self, input_dim=256, output_dim=1, n_ups=2):
        super(Forecaster, self).__init__()
        model = []
        ch = input_dim
        for i in range(n_ups):
            model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
            model += [nn.Conv2d(ch, ch // 2, 3, 1, 1)]
            model += [get_norm_layer(ch // 2)]
            model += [nn.ReLU()]
            ch = ch // 2
        model += [nn.Conv2d(ch, output_dim, 3, 1, 1)]
        # model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        out = self.model(x)
        return out
####################### Vid-ODE model ########################################
class VidODE(nn.Module):
    def __init__(self, input_size=cfg.VidODE.INPUT_SIZE,
                 input_dim = cfg.VidODE.INPUT_DIM,
                 init_dim=cfg.VidODE.INIT_DIM,
                 n_downs=cfg.VidODE.NUM_DOWNSAMPLING,
                 n_layers = cfg.VidODE.NUM_LAYERS,
                 device = cfg.GLOBAL.DEVICE):
        super(VidODE, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size
        self.init_dim = init_dim
        self.n_downs = n_downs
        self.n_layers = n_layers
        self.device = device
        # channels for encoder, ODE, init decoder
        init_dim = self.init_dim  # 64
        resize = 2 ** self.n_downs # 2**3 = 8
        base_dim = init_dim * resize # 64 * 8 = 512
        input_size = (self.input_size // resize, self.input_size // resize) # (8, 8)
        ode_dim = base_dim  # 512
        #### Encoder
        ### Conv Encoder: nn.Sequential
        self.encoder_conv = Encoder(input_dim=self.input_dim, # 1
                                    ch=init_dim,  # 64
                                    n_downs=self.n_downs).to(self.device)

        ### ODE Encoder
        ## requirements: ode_func, ode_solver, ...
        # ode_func_netR: nn.Sequential
        ode_func_netE = make_convnet(n_inputs=ode_dim,  # 512
                                       n_outputs=base_dim, # 512
                                       n_layers=self.n_layers,
                                       n_units=base_dim // 2).to(self.device)
        # ode_func: nn.Module
        rec_ode_func = ODEFunc(input_dim=ode_dim,
                               latent_dim=base_dim,  # channels after encoder, & latent dimension
                               ode_func_net=ode_func_netE).to(self.device)
        # ode_solver: nn.Module
        z0_diffeq_solver = DiffeqSolver(base_dim,
                                        ode_func=rec_ode_func,
                                        method="euler",
                                        latents=base_dim, # 512
                                        odeint_rtol=1e-3,
                                        odeint_atol=1e-4,
                                        device=self.device)

        # encoder_ode_convGRU: nn.Module
        self.encoder_z0 = Encoder_z0_ODE_ConvGRU(input_size=input_size,
                                                 input_dim=base_dim,
                                                 hidden_dim=base_dim,
                                                 kernel_size=(3, 3),
                                                 num_layers=self.n_layers,
                                                 dtype=torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor,
                                                 batch_first=True,
                                                 bias=True,
                                                 return_all_layers=True,
                                                 z0_diffeq_solver=z0_diffeq_solver,
                                                 run_backwards=cfg.VidODE.RUN_BACKWARDS).to(self.device)

        #### Decoder
        ### ODE Decoder
        ode_func_netD = make_convnet(n_inputs=ode_dim,
                                       n_outputs=base_dim,
                                       n_layers=self.n_layers,
                                       n_units=base_dim // 2).to(self.device)

        gen_ode_func = ODEFunc(input_dim=ode_dim,
                               latent_dim=base_dim,
                               ode_func_net=ode_func_netD).to(self.device)

        self.diffeq_solver = DiffeqSolver(base_dim,
                                          gen_ode_func,
                                          cfg.MODEL.ODE_METHODS, base_dim,
                                          odeint_rtol=1e-3,
                                          odeint_atol=1e-4,
                                          device=self.device)
        ### Conv Decoder
        self.decoder_conv = Forecaster(input_dim=base_dim * 2, output_dim=self.input_dim + 3, n_ups=self.n_downs).to(self.device)

    def forward(self, time_steps_to_predict, truth, truth_time_steps, mask=None, out_mask=None):
        resize = 2 ** self.n_downs
        ### gt: b x t x c x h x w
        b, t, c, h, w = truth.shape
        pred_t_len = len(time_steps_to_predict)
        ##### Skip connection forwarding
        skip_image = truth[:,-1, ...]
        skip_conn_embed = self.encoder_conv(skip_image).view(b, -1, h // resize, w // resize)
        #### Encoding
        ### Conv encoding
        e_truth = self.encoder_conv(
            truth.reshape(b * t, c, h, w)).reshape(b, t, -1, h // resize, w // resize)
        ### input_tensor: B*S x C x H x W
        first_point_mu, first_point_std = self.encoder_z0(input_tensor=e_truth,
                                                          time_steps=truth_time_steps,
                                                          mask=mask)
        # Sampling latent features
        first_point_enc = first_point_mu.unsqueeze(0).repeat(1, 1, 1, 1, 1)
        #### Decoding
        ### ODE decoding
        first_point_enc = first_point_enc.squeeze(0)
        sol_y = self.diffeq_solver(time_steps_to_predict, first_point_enc)
        ### Conv decoding
        ## pred_t_len = len(time_steps_to_predict)
        sol_y = sol_y.contiguous().view(b, pred_t_len, -1, h // resize, w // resize)
        # regular b, t, 6, h, w / irregular b, t * ratio, 6, h, w
        pred_outputs = self.get_flowmaps(sol_out=sol_y,
                                         first_prev_embed=skip_conn_embed,
                                         mask=out_mask)  # b, t, 6, h, w
        pred_outputs = torch.cat(pred_outputs, dim=1)
        pred_flows, pred_intermediates, pred_masks = \
            pred_outputs[:, :, :2, ...], pred_outputs[:, :, 2:2 + self.input_dim, ...], \
            torch.sigmoid(pred_outputs[:, :, 2 + self.input_dim:, ...])

        ### Warping first frame by using optical flow
        # Declare grid for warping
        grid_x = torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1).expand(b, h, -1, -1)
        grid_y = torch.linspace(-1.0, 1.0, h).view(1, h, 1, 1).expand(b, -1, w, -1)
        grid = torch.cat([grid_x, grid_y], 3).float().to(self.device)  # [b, h, w, 2]
        # Warping
        last_frame = truth[:, -1, ...]
        warped_pred_x = self.get_warped_images(pred_flows=pred_flows, start_image=last_frame, grid=grid)
        # regular b, t, 6, h, w / irregular b, t * ratio, 6, h, w
        warped_pred_x = torch.cat(warped_pred_x, dim=1)
        ###------------------------------------------------------------------------
        pred_x = pred_masks * warped_pred_x + (1 - pred_masks) * pred_intermediates
        ###------------------------------------------------------------------------
        #### Final result: B x S x C x H x W
        pred_x = pred_x.view(b, -1, c, h, w)
        ### Modify---------------------------------------------------------
        pred_x = torch.clip(pred_x, 0.0, 1.0)
        pred_x = torch.permute(pred_x, (1, 0, 2, 3, 4))
        pred_intermediates = torch.permute(pred_intermediates, (1, 0, 2, 3, 4))
        return pred_x, pred_intermediates

    def get_flowmaps(self, sol_out, first_prev_embed, mask):
        """ Get flowmaps recursively
        Input:
            sol_out - Latents from ODE decoder solver (b, time_steps_to_predict, c, h, w)
            first_prev_embed - Latents of last frame (b, c, h, w)
        Output:
            pred_flows - List of predicted flowmaps (b, time_steps_to_predict, c, h, w)
        """
        b, t, c, h, w = sol_out.size()
        pred_time_steps = int(t)
        pred_flows = list()

        prev = first_prev_embed.clone()
        time_iter = range(pred_time_steps)

        if mask is not None:
            if mask.size(1) == sol_out.size(1):
                sol_out = sol_out[mask.squeeze(-1).byte()].view(b, pred_time_steps, c, h, w)

        for t in time_iter:
            cur_and_prev = torch.cat([sol_out[:, t, ...], prev], dim=1)
            pred_flow = self.decoder_conv(cur_and_prev).unsqueeze(1)
            pred_flows += [pred_flow]
            prev = sol_out[:, t, ...].clone()
        return pred_flows

    def get_warped_images(self, pred_flows, start_image, grid):
        """ Get warped images recursively
        Input:
            pred_flows - Predicted flowmaps to use (b, time_steps_to_predict, c, h, w)
            start_image- Start image to warp
            grid - pre-defined grid
        Output:
            pred_x - List of warped (b, time_steps_to_predict, c, h, w)
        """
        warped_time_steps = pred_flows.size(1)
        pred_x = list()
        last_frame = start_image
        b, _, c, h, w = pred_flows.shape

        for t in range(warped_time_steps):
            pred_flow = pred_flows[:, t, ...]  # b, 2, h, w
            pred_flow = torch.cat(
                [pred_flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), pred_flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
            pred_flow = pred_flow.permute(0, 2, 3, 1)  # b, h, w, 2
            flow_grid = grid.clone() + pred_flow.clone()  # b, h, w, 2
            warped_x = nn.functional.grid_sample(last_frame, flow_grid, padding_mode="border")
            pred_x += [warped_x.unsqueeze(1)]  # b, 1, 3, h, w
            last_frame = warped_x.clone()
        return pred_x

"""
#### Testing
input_data = torch.ones(size=(3, 4, 1, 64, 64)).to(torch.device("cuda"))
input_data = torch.permute(input_data, (1,0,2,3,4))
t_en = np.array([0, 1, 2], dtype=float)
t_fo = np.array([3, 4, 5], dtype=float)
time_step_encoder = torch.from_numpy(t_en).to(torch.device("cuda"))
time_step_forecaster = torch.from_numpy(t_fo).to(torch.device("cuda"))
VidODE_model = VidODE()

pred_data, pred_diff = VidODE_model(time_step_forecaster,
                   input_data,
                   time_step_encoder)

### View params
#count_parameters(VidODE_model)
#print("memory_reserved: {}".format(torch.cuda.memory_reserved(0)))
"""


