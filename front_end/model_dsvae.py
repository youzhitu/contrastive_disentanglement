""" DSVAE """

from collections import OrderedDict
import torch
import torch.nn as nn
from front_end.model_misc import conv1d_unit, mlp
from front_end.model_res2net import Res2Block
from front_end.model_res2net import Res2Net
from front_end.feat_proc import FeatureExtractionLayer


class DSVAE(nn.Module):
    """ referring to https://github.com/yatindandi/Disentangled-Sequential-Autoencoder and
     https://github.com/JunwenBai/C-DSVAE """
    def __init__(self, feat_dim=80, filters='512-512-512-512-1536', kernel_sizes='5-3-3-3-1', dilations='1-2-3-4-1',
                 pooling='ctdstats-128-1', embedding_dims='192', n_shared_layers=4, content_hidden_dim=512,
                 content_emb_dim=16, dec_filters='512-80', dec_kernel_sizes='3-5', dec_dilations='2-1', spec_aug=False,
                 ds_factor=1, name='dsvae'):
        super().__init__()

        self.feat_dim = feat_dim
        self.n_shared_layers = n_shared_layers
        self.shared_enc_out_dim = int(filters.split('-')[n_shared_layers - 1])
        self.spk_emb_dim = int(embedding_dims.split('-')[-1])
        self.content_hidden_dim = content_hidden_dim
        self.content_emb_dim = content_emb_dim
        self.dec_filters = [int(filter_) for filter_ in dec_filters.split('-')]
        self.dec_kernel_sizes = [int(kernel_size_) for kernel_size_ in dec_kernel_sizes.split('-')]
        self.dec_dilations = [int(dilation_) for dilation_ in dec_dilations.split('-')]
        self.ds_factor = ds_factor
        self.name = name

        # Create layers
        self.spk_encoder = SpeakerEncoder(
            n_shared_layers=n_shared_layers, feat_dim=feat_dim, filters=filters, kernel_sizes=kernel_sizes,
            dilations=dilations, pooling=pooling, embedding_dims=embedding_dims, spec_aug=spec_aug)

        self.content_encoder = ContentEncoder(
            input_dim=self.shared_enc_out_dim, hidden_dim=content_hidden_dim, emb_dim=content_emb_dim,
            ds_factor=self.ds_factor)

        self.decoder = self.create_decoder(input_dim=self.spk_emb_dim + content_emb_dim)
        self.content_prior_encoder = ContentPriorEncoder(hidden_dim=content_hidden_dim, emb_dim=content_emb_dim)

        self.mi_feat_in_spk_emb = MIInfoNCE(input_dims=[feat_dim, self.spk_emb_dim])
        # self.mi_feat_in_spk_emb = MIInfoNCESeqVec(input_dims=[self.spk_emb_dim, feat_dim])
        self.mi_feat_in_content_emb = MIInfoNCESeqVec(input_dims=[feat_dim, content_emb_dim])
        # self.mi_feat_in_content_emb = MIInfoNCESeqSeq(input_dims=[self.spk_emb_dim, feat_dim])
        self.mi_content_emb_spk_emb = MIInfoNCE(input_dims=[content_emb_dim, self.spk_emb_dim])

    def create_decoder(self, input_dim=256):
        dec_layers = OrderedDict()

        dec_layers['conv0'] = conv1d_unit(
            input_dim, self.dec_filters[0], self.dec_kernel_sizes[0], padding=2, dilation=self.dec_dilations[0],
            transpose=True)

        for i in range(1, len(self.dec_filters) - 1):
            dec_layers[f'conv{i}'] = conv1d_unit(
                self.dec_filters[i - 1], self.dec_filters[i], self.dec_kernel_sizes[i], dilation=self.dec_dilations[i],
                transpose=True)

        dec_layers[f'conv{len(self.dec_filters) - 1}'] = conv1d_unit(
            self.dec_filters[-2], self.dec_filters[-1], self.dec_kernel_sizes[-1], padding=2,
            dilation=self.dec_dilations[-1], act=None, transpose=True)

        return nn.Sequential(dec_layers)

    def upsample_content(self, z_content, x_feat):
        if self.ds_factor > 1:
            z_content = z_content.repeat(1, self.ds_factor, 1)

            if z_content.size(1) > x_feat.size(2):
                z_content = z_content[:, :x_feat.size(2), :]

        return z_content

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, seq_len]
        Returns:
            z_spk, z_spk_mean, z_spk_logvar: Tensor, [batch_size, spk_embedding_dim]
            z_content, z_content_mean, z_content_logvar: Tensor, [batch_size, seq_len, content_embedding_dim]
            x_dec: Tensor, [batch_size, input_dim, seq_len]
        """

        z_spk, z_spk_mean, z_spk_logvar, x_share, x_feat = self.spk_encoder(x, sampling=self.training)
        z_content, z_content_mean, z_content_logvar = self.content_encoder(x_share, sampling=self.training)

        z_spk_expand = z_spk.unsqueeze(1).expand(-1, x_feat.size(2), -1)
        z_content = self.upsample_content(z_content, x_feat)
        z_content_mean = self.upsample_content(z_content_mean, x_feat)
        z_content_logvar = self.upsample_content(z_content_logvar, x_feat)

        x_dec = self.decoder(torch.cat([z_content, z_spk_expand], dim=2).transpose(1, 2))

        z_content_prior, z_content_prior_mean, z_content_prior_logvar = self.content_prior_encoder(z_content)

        mi_feat_in_spk_emb = self.mi_feat_in_spk_emb(torch.mean(x_feat, dim=2), z_spk)
        mi_feat_in_content_emb = self.mi_feat_in_content_emb(x_feat, torch.mean(z_content, dim=1))
        mi_content_emb_spk_emb = self.mi_content_emb_spk_emb(torch.mean(z_content, dim=1), z_spk)

        return z_spk, z_spk_mean, z_spk_logvar, z_content, z_content_mean, z_content_logvar, \
            x_dec, x_feat, z_content_prior, z_content_prior_mean, z_content_prior_logvar, \
            mi_feat_in_spk_emb, mi_feat_in_content_emb, mi_content_emb_spk_emb


class SpeakerEncoder(Res2Net):
    def __init__(self, n_shared_layers=3, **kwargs):
        self.n_shared_layers = n_shared_layers
        super().__init__(**kwargs)

        self.spk_model = None
        self.feat_layer = FeatureExtractionLayer(feat_dim=self.feat_dim, spec_aug=self.spec_aug)
        self.frame_layer = self.create_frame_level_layers(input_dim=self.feat_dim)
        self.pool_layer = self.create_pooling_layer(input_dim=self.filters[-1])
        self.emb_layer = self.create_emb_layers(input_dim=self.pool_layer[0].output_dim)
        self.emb_logvar = self.create_emb_layers(input_dim=self.pool_layer[0].output_dim)

    def create_frame_level_layers(self, input_dim=80):
        return Res2FrameLevelLayers(
            input_dim, self.filters, self.kernel_sizes, self.dilations, scale=self.scale, se=self.se,
            se_hidden_dim=self.se_hidden_dim, n_shared_layers=self.n_shared_layers)

    def spk_encoder_forward(self, x, sampling=True):
        x_feat = self.feat_layer(x)
        x, x_share = self.frame_layer(x_feat)
        x = self.pool_layer(x)
        z_mean = self.emb_layer(x)
        z_logvar = self.emb_logvar(x)
        z = reparameterize(z_mean, z_logvar, sampling=sampling)

        return z, z_mean, z_logvar, x_share, x_feat

    def forward(self, x, sampling=True):
        """
        Args:
            x: Tensor, [batch_size, seq_len]
            sampling: bool
        Returns:
            z, z_mean, z_logvar: Tensor, [batch_size, embedding_dim]
            x_share: Tensor, [batch_size, hidden_dim, seq_len]
        """

        z, z_mean, z_logvar, x_share, x_feat = self.spk_encoder_forward(x, sampling=sampling)

        return z, z_mean, z_logvar, x_share, x_feat


class Res2FrameLevelLayers(nn.Module):
    def __init__(self, input_dim, filters, kernel_sizes, dilations, scale=8, se=True, se_hidden_dim=128,
                 n_shared_layers=3):
        super().__init__()

        self.n_shared_layers = n_shared_layers
        self.frame_level_layers = nn.ModuleList()
        self.frame_level_layers.append(
            conv1d_unit(input_dim, filters[0], kernel_sizes[0], padding='same', dilation=dilations[0]))

        for i in range(1, len(filters) - 1):
            self.frame_level_layers.append(
                Res2Block(filters[i - 1], filters[i], kernel_sizes[i], dilation=dilations[i],
                          scale=scale, se=se, se_hidden_dim=se_hidden_dim))

        self.frame_level_layers.append(
            conv1d_unit(sum(filters[1:-1]), filters[-1], kernel_sizes[-1], padding='same', dilation=dilations[-1]))

        self.output_chnls = filters[-1]  # as pooling input dim

    def forward(self, x):
        x = self.frame_level_layers[0](x)
        x_res2blk0 = self.frame_level_layers[1](x)
        x_res2blk1 = self.frame_level_layers[2](x + x_res2blk0)
        x_res2blk2 = self.frame_level_layers[3](x + x_res2blk0 + x_res2blk1)
        x_out = self.frame_level_layers[4](torch.cat([x_res2blk0, x_res2blk1, x_res2blk2], dim=1))

        if self.n_shared_layers == 5:
            x_share = x_out
        elif self.n_shared_layers == 4:
            x_share = x_res2blk2
        elif self.n_shared_layers == 3:
            x_share = x_res2blk1
        elif self.n_shared_layers == 2:
            x_share = x_res2blk0
        else:
            x_share = x

        return x_out, x_share


class ContentEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, emb_dim=16, ds_factor=1):
        super().__init__()

        self.content_enc_lstm = nn.LSTM(input_dim, hidden_dim, 1, bidirectional=True, batch_first=True)
        self.content_enc_rnn = nn.RNN(hidden_dim * 2, hidden_dim, 1, batch_first=True)
        self.content_emb_mean = nn.Linear(hidden_dim, emb_dim)
        self.content_emb_logvar = nn.Linear(hidden_dim, emb_dim)
        self.ds_factor = ds_factor

    def forward(self, x, sampling=True):
        """
        Args:
            x: Tensor, [batch_size, input_dim, seq_len]
            sampling: bool
        Returns:
            z, z_mean, z_logvar: Tensor, [batch_size, seq_len, embedding_dim]
        """

        x = x.transpose(1, 2)
        x, _ = self.content_enc_lstm(x)
        x, _ = self.content_enc_rnn(x)
        z_mean = self.content_emb_mean(x)
        z_logvar = self.content_emb_logvar(x)
        z = reparameterize(z_mean, z_logvar, sampling=sampling)

        return z[:, ::self.ds_factor, :], z_mean[:, ::self.ds_factor, :], z_logvar[:, ::self.ds_factor, :]


class ContentPriorEncoder(nn.Module):
    def __init__(self, hidden_dim=512, emb_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.content_prior_enc = nn.LSTMCell(emb_dim, hidden_dim)
        self.content_prior_mean = nn.Linear(hidden_dim, emb_dim)
        self.content_prior_logvar = nn.Linear(hidden_dim, emb_dim)

    def forward(self, z_content_post, sampling=True):
        batch_size, n_frames, _ = z_content_post.size()
        z_content_prior, z_content_prior_mean, z_content_prior_logvar = None, None, None

        z_content_prior_t = torch.zeros(batch_size, self.emb_dim, device=z_content_post.device)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=z_content_post.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=z_content_post.device)

        for _ in range(n_frames):
            h_t, c_t = self.content_prior_enc(z_content_prior_t, (h_t, c_t))
            z_content_prior_mean_t = self.content_prior_mean(h_t)
            z_content_prior_logvar_t = self.content_prior_logvar(h_t)
            z_content_prior_t = reparameterize(z_content_prior_mean_t, z_content_prior_logvar_t, sampling)

            z_content_prior_tmp = z_content_prior_t.unsqueeze(1)  # [batch_size, 1, emb_dim]
            z_content_prior_mean_tmp = z_content_prior_mean_t.unsqueeze(1)
            z_content_prior_logvar_tmp = z_content_prior_logvar_t.unsqueeze(1)

            if z_content_prior is None:
                z_content_prior = z_content_prior_tmp
                z_content_prior_mean = z_content_prior_mean_tmp
                z_content_prior_logvar = z_content_prior_logvar_tmp
            else:
                z_content_prior = torch.cat((z_content_prior, z_content_prior_tmp), dim=1)
                z_content_prior_mean = torch.cat((z_content_prior_mean, z_content_prior_mean_tmp), dim=1)
                z_content_prior_logvar = torch.cat((z_content_prior_logvar, z_content_prior_logvar_tmp), dim=1)

        return z_content_prior, z_content_prior_mean, z_content_prior_logvar


class MIInfoNCE(nn.Module):
    """ MI estimator between two embedding vectors using an InfoNCE lower bound,
    https://colab.research.google.com/github/google-research/google-research/blob/master/vbmi/vbmi_demo.ipynb """

    def __init__(self, input_dims, hidden_dims='128-64'):
        super().__init__()
        hidden_dims = [int(hidden_dim) for hidden_dim in hidden_dims.split('-')]
        self.mlp_g = mlp(
            input_dim=input_dims[0], fc_dims=hidden_dims, norms=[None] * len(hidden_dims), acts=[nn.ReLU(), None])
        self.mlp_h = mlp(
            input_dim=input_dims[1], fc_dims=hidden_dims, norms=[None] * len(hidden_dims), acts=[nn.ReLU(), None])

    def forward(self, x, y):
        """
        Args:
            x: Tensor, [batch_size, input_dim0]
            y: Tensor, [batch_size, input_dim1]
        Returns:
            mi: float
        """

        x = self.mlp_g(x)
        y = self.mlp_h(y)

        return compute_mi(x, y)


class MIInfoNCESeqVec(nn.Module):
    """ MI estimator between a sequence of vectors and a vector """

    def __init__(self, input_dims, hidden_dims='128-64'):
        super().__init__()
        hidden_dims = [int(hidden_dim) for hidden_dim in hidden_dims.split('-')]
        self.mlp_g = nn.Sequential(
            conv1d_unit(input_dims[0], hidden_dims[0], 1, norm=None, act=nn.ReLU()),
            conv1d_unit(hidden_dims[0], hidden_dims[1], 1, norm=None, act=None))

        self.mlp_h = mlp(input_dim=input_dims[1], fc_dims=hidden_dims, norms=[None] * len(hidden_dims),
                         acts=[nn.ReLU(), None])

    def forward(self, x, y):
        """
        Args:
            x: Tensor, [batch_size, input_dim0, seq_len]
            y: Tensor, [batch_size, input_dim1]
        Returns:
            mi: float
        """

        x = self.mlp_g(x)
        y = self.mlp_h(y)

        infonce = 0.

        for i in range(x.shape[2]):
            infonce += compute_mi(x[:, :, i], y)

        return infonce / x.shape[2]


class MIInfoNCESeqSeq(nn.Module):
    """ MI estimator between a sequence of vectors and a sequence of vectors """

    def __init__(self, input_dims, hidden_dims='128-64'):
        super().__init__()
        hidden_dims = [int(hidden_dim) for hidden_dim in hidden_dims.split('-')]
        self.mlp_g = nn.Sequential(
            conv1d_unit(input_dims[0], hidden_dims[0], 1, norm=None, act=nn.ReLU()),
            conv1d_unit(hidden_dims[0], hidden_dims[1], 1, norm=None, act=None))

        self.mlp_h = nn.Sequential(
            conv1d_unit(input_dims[1], hidden_dims[0], 1, norm=None, act=nn.ReLU()),
            conv1d_unit(hidden_dims[0], hidden_dims[1], 1, norm=None, act=None))

    def forward(self, x, y):
        """
        Args:
            x: Tensor, [batch_size, input_dim0, seq_len]
            y: Tensor, [batch_size, input_dim1, seq_len]
        Returns:
            mi: float
        """

        x = self.mlp_g(x)
        y = self.mlp_h(y)

        infonce = 0.

        for i in range(x.shape[2]):
            for j in range(y.shape[2]):
                infonce += compute_mi(x[:, :, i], y[:, :, j])

        return infonce / (x.shape[2] * y.shape[2])


def reparameterize(mean, logvar, sampling=True):
    return mean + torch.randn_like(logvar).clamp(max=0.01) * torch.exp(0.5 * logvar) if sampling else mean


def compute_mi(x, y):
    """
    Args:
        x, y: Tensor, [batch_size, emb_dim]
    Returns:
        mi: float
    """

    scores = torch.matmul(y, x.T)  # h(y) * g(x)^T
    nll = torch.mean(torch.diag(scores) - torch.logsumexp(scores, dim=1))
    infonce = torch.log(torch.tensor(scores.size(0))) + nll

    return infonce
