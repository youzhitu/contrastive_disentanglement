""" MoCo trainer """

import torch
from front_end.trainer import Trainer


class TrainerMoCo(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model_out, epoch=None):
        logits, label = model_out  # logits: [B, 1 + K]
        loss = self.loss_fn(logits, label)
        # loss = torch.mean(torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0])
        reg_loss = self.weight_decay * sum([(para ** 2).sum() for para in self.model.parameters()])

        return loss + reg_loss, loss.view(1,)
