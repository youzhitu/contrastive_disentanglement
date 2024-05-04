""" MoCo """

from collections import OrderedDict
import torch.nn as nn
import torch
from front_end.model_res2net import Res2Net
from front_end.model_misc import dist_concat_all_gather
import torch.distributed as dist


class MoCo(Res2Net):
    """ referring to https://github.com/facebookresearch/moco/blob/main/moco/builder.py """
    def __init__(self, moco_cfg='65536-0.999-0.05', name='moco', **kwargs):
        moco_cfg = moco_cfg.split('-')
        self.K = int(moco_cfg[0])  # queue size
        self.m = float(moco_cfg[1])  # momentum
        self.T = float(moco_cfg[2])  # temperature
        super().__init__(name=name, **kwargs)

        # create the encoders
        self.spk_model_k = nn.Sequential(OrderedDict([('spk_encoder', self.create_spk_encoder())]))

        for param_q, param_k in zip(self.spk_model.parameters(), self.spk_model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer('queue', torch.randn(self.embedding_dims[-1], self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """

        for param_q, param_k in zip(self.spk_model.parameters(), self.spk_model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = dist_concat_all_gather(keys)  # gather keys before updating queue

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = dist_concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        dist.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = dist_concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, x_q, x_k):
        """
        Input:
            x_q: queries, [B, seq_len]
            x_k: keys, [B, seq_len]
        Output:
            logits, labels: [B, 1 + K]
        """

        # compute query features
        q = self.spk_model(x_q)  # queries: [B, C]
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            x_k, idx_unshuffle = self._batch_shuffle_ddp(x_k)  # shuffle for making use of BN

            k = self.spk_model_k(x_k)  # keys: [B, C]
            k = nn.functional.normalize(k, dim=1)

            k = self._batch_unshuffle_ddp(k, idx_unshuffle)  # undo shuffle

        # compute positive and negative logits, einstein sum is more intuitive
        l_pos = torch.einsum('bc,bc->b', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('bc,ck->bk', [q, self.queue.clone().detach()])  # [B, K]

        logits = torch.cat([l_pos, l_neg], dim=1) / self.T  # [B, 1 + K]
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()  # positive key indicators

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
