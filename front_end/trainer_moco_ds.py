""" MoCoDS trainer """

from front_end.trainer import Trainer
import torch
import torch.nn as nn


class TrainerMoCoDS(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.mse = nn.MSELoss(reduction='mean')

    def compute_loss(self, model_out, epoch=None):
        logits, labels, z_spk, z_spk_mean, z_spk_logvar, z_content, z_content_mean, z_content_logvar, \
            x_dec, x_feat, z_content_prior, z_content_prior_mean, z_content_prior_logvar, \
            mi_feat_in_spk_emb, mi_feat_in_content_emb, mi_content_emb_spk_emb, \
            z_spk_k, z_spk_mean_k, z_spk_logvar_k, z_content_k, z_content_mean_k, z_content_logvar_k, \
            x_dec_k, x_feat_k, z_content_prior_k, z_content_prior_mean_k, z_content_prior_logvar_k, \
            mi_feat_in_spk_emb_k, mi_feat_in_content_emb_k, mi_content_emb_spk_emb_k = model_out

        moco_loss = self.loss_fn(logits, labels)
        reg_loss = self.weight_decay * sum([(para ** 2).sum() for para in self.model.parameters()])
        mse = (self.mse(x_dec, x_feat) + self.mse(x_dec_k, x_feat_k)) / 2

        batch_size = z_spk.size(0)
        kld_spk = 0.5 * torch.sum(torch.pow(z_spk_mean, 2) + torch.exp(z_spk_logvar) - 1 - z_spk_logvar) / batch_size
        kld_content = 0.5 * torch.sum(
            (torch.pow(z_content_mean - z_content_prior_mean, 2) + torch.exp(z_content_logvar)) /
            torch.exp(z_content_prior_logvar) + z_content_prior_logvar - z_content_logvar - 1) / batch_size

        kld_spk_k = 0.5 * torch.sum(torch.pow(z_spk_mean_k, 2) + torch.exp(z_spk_logvar_k) - 1 - z_spk_logvar_k) \
            / batch_size
        kld_content_k = 0.5 * torch.sum(
            (torch.pow(z_content_mean_k - z_content_prior_mean_k, 2) + torch.exp(z_content_logvar_k)) /
            torch.exp(z_content_prior_logvar_k) + z_content_prior_logvar_k - z_content_logvar_k - 1) / batch_size

        dsvae_loss = mse + 1. * (kld_spk + 1. * kld_content - 1. * (mi_feat_in_spk_emb + mi_feat_in_content_emb) +
                                 1. * mi_content_emb_spk_emb) / 2 + \
            1. * (kld_spk_k + 1. * kld_content_k - 1. * (mi_feat_in_spk_emb_k + mi_feat_in_content_emb_k) +
                  1. * mi_content_emb_spk_emb_k) / 2

        return moco_loss + reg_loss + 0.01 * dsvae_loss, \
            torch.tensor([moco_loss, dsvae_loss, mse, kld_spk, kld_content, mi_content_emb_spk_emb])
