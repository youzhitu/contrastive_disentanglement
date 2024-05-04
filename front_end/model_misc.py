""" Sub-models """

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.distributed as dist


def conv1d_unit(in_chanls, out_chanls, kernel_size, stride=1, padding=0, dilation=1, groups=1, act=nn.GELU(),
                norm='bn', post_norm=True, transpose=False):
    conv1d_unit_layers = OrderedDict()

    if transpose:
        conv1d_unit_layers['conv'] = nn.ConvTranspose1d(
            in_chanls, out_chanls, kernel_size, stride=stride, padding=padding, dilation=dilation)
    else:
        conv1d_unit_layers['conv'] = nn.Conv1d(
            in_chanls, out_chanls, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

    if post_norm:
        if act is not None:
            conv1d_unit_layers['act'] = act

        if norm == 'bn':
            conv1d_unit_layers['norm'] = nn.BatchNorm1d(out_chanls)
        else:
            conv1d_unit_layers['norm'] = nn.Identity()
    else:
        if norm == 'bn':
            conv1d_unit_layers['norm'] = nn.BatchNorm1d(out_chanls)
        else:
            conv1d_unit_layers['norm'] = nn.Identity()

        if act is not None:
            conv1d_unit_layers['act'] = act

    return nn.Sequential(conv1d_unit_layers)


def linear_unit(in_nodes, out_nodes, act=nn.GELU(), norm='bn', post_norm=True):
    linear_unit_layers = OrderedDict()
    linear_unit_layers['linear'] = nn.Linear(in_nodes, out_nodes)

    if post_norm:
        if act is not None:
            linear_unit_layers['act'] = act

        if norm == 'bn':
            linear_unit_layers['norm'] = nn.BatchNorm1d(out_nodes)
        else:
            linear_unit_layers['norm'] = nn.Identity()
    else:
        if norm == 'bn':
            linear_unit_layers['norm'] = nn.BatchNorm1d(out_nodes)
        else:
            linear_unit_layers['norm'] = nn.Identity()

        if act is not None:
            linear_unit_layers['act'] = act

    return nn.Sequential(linear_unit_layers)


def mlp(input_dim, fc_dims, norms, acts, post_norm=True):
    fc_layers = OrderedDict()

    if len(fc_dims) > 1:
        fc_layers['fc0'] = linear_unit(input_dim, fc_dims[0], norm=norms[0], act=acts[0], post_norm=post_norm)
        input_dim = fc_dims[0]

        for i in range(1, len(fc_dims) - 1):
            fc_layers[f'fc{i}'] = linear_unit(
                input_dim, fc_dims[i], norm=norms[i], act=acts[i], post_norm=post_norm)
            input_dim = fc_dims[i]

    fc_layers[f'fc{len(fc_dims) - 1}'] = linear_unit(
        input_dim, fc_dims[-1], norm=norms[-1], act=acts[-1], post_norm=post_norm)

    return nn.Sequential(fc_layers)


class SEBlock(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        # noinspection PyTypeChecker
        self.se = nn.Sequential(OrderedDict(
            [('gap', nn.AdaptiveAvgPool1d(1)),
             ('fc1', conv1d_unit(input_dim, hidden_dim, kernel_size=1, act=nn.ReLU(), norm=None)),
             ('fc2', conv1d_unit(hidden_dim, input_dim, kernel_size=1, act=nn.Sigmoid(), norm=None))
             ]))

    def forward(self, x):
        return self.se(x) * x


@torch.no_grad()
def dist_concat_all_gather(input_tensor):
    """
    Performs distributed concatenate using all_gather operation on the provided tensors
    *** Warning ***: dist.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(input_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, input_tensor, async_op=False)

    return torch.cat(tensors_gather, dim=0)
