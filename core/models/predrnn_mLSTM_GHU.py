__author__ = 'yunbo'

import torch
import torch.nn as nn
from core.layers.CausalmLSTMCell import CausalmLSTMCell
from core.layers.GraddientHighwayUnit import GHU

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                CausalmLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        
        # Initialize Gradient Highway Unit (GHU)
        self.ghu = GHU(
            layer_name='highway',
            filter_size=configs.filter_size,
            num_features=num_hidden[0],
            input_channels=num_hidden[0],  # Matches output of first CausalLSTM
            tln=configs.layer_norm
        )

    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        states = []
        z_t = None

        for i in range(self.num_layers):
            states.append(self.cell_list[i].init_states((batch, height, width, self.frame_channel)))

        for t in range(self.configs.total_length - 1):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.input_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            # H_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            H_t, states[0]  = self.cell_list[0](net, states[0])
            
            # Apply Gradient Highway Unit (GHU)
            z_t = self.ghu(H_t, z_t)  # GHU updates its state

            # Process second layer: CausalLSTM with GHU output
            H_t, states[1] = self.cell_list[1](z_t, states[1])


            for i in range(2, self.num_layers):
                # h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                H_t, states[i] = self.cell_list[i](H_t, states[i])
                # print(H_t.shape)

            x_gen = self.conv_last(H_t.permute(0, 3, 1, 2))
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        return next_frames, loss
