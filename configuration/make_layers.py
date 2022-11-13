import torch.nn as nn
import torch
from collections import  OrderedDict

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'bn' in layer_name:
                batch_norm = nn.BatchNorm2d(num_features=v[1])
                layers.append(('bn_' + layer_name, batch_norm))
            elif 'in' in layer_name:
                instance_norm = nn.InstanceNorm2d(num_features=v[1])
                layers.append(('in_' + layer_name, instance_norm))

            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            elif 'gelu' in layer_name:
                layers.append(('gelu_' + layer_name, nn.GELU()))
            elif 'prelu' in layer_name:
                layers.append(('prelu_' + layer_name, nn.PReLU()))
            elif 'tanh' in layer_name:
                layers.append(('tanh_' + layer_name, nn.Tanh()))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'bn' in layer_name:
                batch_norm = nn.BatchNorm2d(num_features=v[1])
                layers.append(('bn_' + layer_name, batch_norm))
            elif 'in' in layer_name:
                instance_norm = nn.InstanceNorm2d(num_features=v[1])
                layers.append(('in_' + layer_name, instance_norm))

            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            elif 'gelu' in layer_name:
                layers.append(('gelu_' + layer_name, nn.GELU()))
            elif 'prelu' in layer_name:
                layers.append(('prelu_' + layer_name, nn.PReLU()))
            elif 'tanh' in layer_name:
                layers.append(('tanh_' + layer_name, nn.Tanh()))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))

def make_convnet(n_inputs, n_outputs, n_layers=1, n_units=128, nonlinear='tanh'):
    if nonlinear == 'tanh':
        nonlinear = nn.Tanh()
    else:
        raise NotImplementedError('There is no named')

    layers = []
    layers.append(nn.Conv2d(n_inputs, n_units, 3, 1, 1, dilation=1))

    for i in range(n_layers):
        layers.append(nonlinear)
        layers.append(nn.Conv2d(n_units, n_units, 3, 1, 1, dilation=1))

    layers.append(nonlinear)
    layers.append(nn.Conv2d(n_units, n_outputs, 3, 1, 1, dilation=1))

    return nn.Sequential(*layers)

class Attention_net(nn.Module):
    def __init__(self, num_filters):
        super(Attention_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        ).to(torch.device("cuda"))
    def forward(self, t, state):
        return self.net(state)