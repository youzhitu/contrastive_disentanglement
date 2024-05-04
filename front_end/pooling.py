""" Pooling layers """

import torch
import torch.nn as nn
from front_end.model_misc import conv1d_unit


class StatsPoolingLayer(nn.Module):
    def __init__(self, input_dim=1536, stats_type='standard', use_mean=True):
        super(StatsPoolingLayer, self).__init__()
        self.stats_type = stats_type
        self.use_mean = use_mean
        self.output_dim = input_dim * 2 if self.use_mean else input_dim

    def forward(self, x):
        if self.stats_type == 'standard':
            std, mean = torch.std_mean(x, dim=-1, unbiased=False)
            return torch.cat((mean, std), dim=1) if self.use_mean else std
        elif self.stats_type == 'rms':
            var, mean = torch.var_mean(x, dim=-1, unbiased=False)
            rms = torch.sqrt(torch.square(mean) + var)
            return torch.cat((mean, rms), dim=1) if self.use_mean else rms
        else:
            raise NotImplementedError


class ChannelContextStatsPoolingLayer(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=256, context_dependent=False):
        super().__init__()
        self.input_dim = input_dim
        self.context_dependent = context_dependent

        if self.context_dependent:
            con1d_1st = conv1d_unit(input_dim * 3, hidden_dim, 1, act=nn.Tanh(), post_norm=False, norm=None)
        else:
            con1d_1st = conv1d_unit(input_dim, hidden_dim, 1, act=nn.Tanh(), post_norm=False, norm=None)

        self.attention_net = nn.Sequential(
            con1d_1st,
            conv1d_unit(hidden_dim, input_dim, 1, act=nn.Softmax(dim=2), norm=None))

        self.output_dim = input_dim * 2

    def forward(self, x):
        if self.context_dependent:
            seq_len = x.size(2)
            x_att = torch.concat([x, torch.mean(x, dim=2, keepdim=True).tile((1, 1, seq_len)),
                                  torch.std(x, dim=2, keepdim=True).tile((1, 1, seq_len))], dim=1)
        else:
            x_att = x

        att_weight = self.attention_net(x_att)
        mean = torch.sum(x * att_weight, dim=2)
        std = torch.sqrt((torch.sum((x ** 2) * att_weight, dim=2) - mean ** 2).clamp(min=1e-5))

        return torch.cat([mean, std], dim=1)
