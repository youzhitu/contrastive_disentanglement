""" X-vector """

from collections import OrderedDict
import torch.nn as nn
from front_end.feat_proc import FeatureExtractionLayer
from front_end.model_misc import conv1d_unit, mlp
from front_end.pooling import StatsPoolingLayer, ChannelContextStatsPoolingLayer


class TDNN(nn.Module):
    def __init__(self, feat_dim=40, filters='512-512-512-512-1536', kernel_sizes='5-3-3-1-1', dilations='1-2-3-1-1',
                 pooling='stats', embedding_dims='512-512', predictor_dims='1024-1024-512', spec_aug=False,
                 name='tdnn'):
        super().__init__()

        self.feat_dim = feat_dim
        self.filters = [int(filter_) for filter_ in filters.split('-')]
        self.kernel_sizes = [int(kernel_size_) for kernel_size_ in kernel_sizes.split('-')]
        self.dilations = [int(dilation_) for dilation_ in dilations.split('-')]

        assert len(self.filters) == len(self.kernel_sizes) == len(self.dilations), \
            'Unequal length of filters, kernel_sizes, or dilation rates!'

        self.pooling = pooling
        self.embedding_dims = [int(embedding_dim) for embedding_dim in embedding_dims.split('-')]
        self.predictor_dims = [int(predictor_dim) for predictor_dim in predictor_dims.split('-')] \
            if name in ['simsiam', 'dino'] else None
        self.spec_aug = spec_aug
        self.name = name

        # Create layers
        self.spk_model = OrderedDict()
        self.spk_model['spk_encoder'] = self.create_spk_encoder()

        if name in ['simsiam', 'dino']:
            self.spk_model['predictor'] = self.create_predictor(input_dim=self.embedding_dims[-1])
        self.spk_model = nn.Sequential(self.spk_model)

    def create_spk_encoder(self):
        spk_enc = OrderedDict()
        spk_enc['feat_layer'] = FeatureExtractionLayer(feat_dim=self.feat_dim, spec_aug=self.spec_aug)
        spk_enc['frame_layer'] = self.create_frame_level_layers(input_dim=self.feat_dim)
        spk_enc['pool_layer'] = self.create_pooling_layer(input_dim=self.filters[-1])
        spk_enc['emb_layer'] = self.create_emb_layers(input_dim=spk_enc['pool_layer'][0].output_dim)

        return nn.Sequential(spk_enc)

    def create_frame_level_layers(self, input_dim=40):
        frame_level_layers = OrderedDict()

        frame_level_layers['conv0'] = conv1d_unit(
            input_dim, self.filters[0], self.kernel_sizes[0], dilation=self.dilations[0])

        for i in range(1, len(self.filters)):
            frame_level_layers[f'conv{i}'] = conv1d_unit(
                self.filters[i - 1], self.filters[i], self.kernel_sizes[i], dilation=self.dilations[i])

        return nn.Sequential(frame_level_layers)

    def create_pooling_layer(self, input_dim=1500):
        pooling_layers = OrderedDict()

        if self.pooling.startswith('ctdstats'):
            hidden_dim, context = list(map(int, self.pooling.split('-')[-2:]))  # ctdstats-128-1
            pooling_layer = ChannelContextStatsPoolingLayer(
                input_dim=input_dim, hidden_dim=hidden_dim, context_dependent=bool(context))
        elif self.pooling == 'stats':
            pooling_layer = StatsPoolingLayer(use_mean=True)
        else:
            raise NotImplementedError

        pooling_layers['pooling'] = pooling_layer
        pooling_layers['bn'] = nn.BatchNorm1d(pooling_layer.output_dim)

        return nn.Sequential(pooling_layers)

    def create_emb_layers(self, input_dim=3000):
        norms, acts = [], []

        if len(self.embedding_dims) > 1:
            norms += ['bn'] * (len(self.embedding_dims) - 1)
            acts += [nn.ReLU()] * (len(self.embedding_dims) - 1)
        norms += [None]
        acts += [None]

        return mlp(input_dim=input_dim, fc_dims=self.embedding_dims, norms=norms, acts=acts)

    def create_predictor(self, input_dim=192):
        norms, acts = [], []

        if len(self.predictor_dims) > 1:
            norms += ['bn'] * (len(self.predictor_dims) - 1)
            acts += [nn.ReLU()] * (len(self.predictor_dims) - 1)
        norms += [None]
        acts += [None]

        return mlp(input_dim=input_dim, fc_dims=self.predictor_dims, norms=norms, acts=acts)

    def forward(self, x1, x2):
        """
        Args:
            x1, x2: Tensor, [batch_size, wav_len]
        Returns:
            p1, p2: Tensor, [batch_size, predictor_dim]
            z1, z2: Tensor, [batch_size, emb_dim]
        """

        z1 = self.spk_model.spk_encoder(x1)
        z2 = self.spk_model.spk_encoder(x2)

        if self.name == 'simsiam':
            p1 = self.spk_model.predictor(z1)
            p2 = self.spk_model.predictor(z2)
            return p1, p2, z1.detach(), z2.detach()
        return z1, z2
