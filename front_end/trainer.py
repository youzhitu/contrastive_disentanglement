""" Base trainer """

from back_end.cosine_score import CosineScorer
from front_end.model_misc import dist_concat_all_gather
import math
import numpy as np
import os
from scipy.linalg import norm
from time import perf_counter
import torch
import torch.distributed as dist
from utils.eval_metrics import eval_performance


class Trainer(object):
    def __init__(self, train_dataloader=None, test_dataloader=None, eval_dataloader=None, model=None, optim='sgd',
                 weight_decay=1e-4, lr='lin_1_cos_1:0.01@0,0.1@3,0.0001@30', epochs=100, device=0, sync_bn=True,
                 ckpt_dir='model_ckpt', ckpt_num=None, save_freq=5, save_ckpts=20, logger=None):

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.optim = optim
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.sync_bn = sync_bn
        self.ckpt_dir = ckpt_dir
        self.ckpt_num = ckpt_num
        self.save_freq = save_freq
        self.save_ckpts = save_ckpts
        self.logger = logger

        # Initialize training
        self.optimizer = None
        self.loss_fn = torch.nn.CosineSimilarity(dim=1)
        self.epochs_trained = 0
        self.lr_scheduler = LRScheduler(self.lr)
        self.train_metrics = Metrics(self.device)
        self.test_metrics = Metrics(self.device) if self.test_dataloader is not None else None
        if not isinstance(train_dataloader.dataset, torch.utils.data.IterableDataset):
            self.batch_size = train_dataloader.batch_sampler.batch_size
        else:
            self.batch_size = train_dataloader.dataset.batch_size

        # Initialize evaluation
        if self.eval_dataloader is not None:
            from utils.my_utils import rd_data_frame

            if 'vox' in self.train_dataloader.dataset.source:
                self.enroll_ids = rd_data_frame('meta/eval/voxceleb1_test_path2info',
                                                ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])['utt_id'].values
                self.test_ids = self.enroll_ids.copy()
                self.trials_file = 'trials/voxceleb1/trials_voxceleb.npz'
            else:
                self.enroll_ids = rd_data_frame('meta/eval/cnceleb1_enroll_path2info',
                                                ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])['utt_id'].values
                self.test_ids = rd_data_frame('meta/eval/cnceleb1_test_path2info',
                                              ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])['utt_id'].values
                self.trials_file = 'trials/cnceleb1/trials.npz'

    def setup_train(self):
        self.setup_model()
        self.setup_optimizer()

        if os.listdir(self.ckpt_dir):
            self.load_checkpoint(map_location=torch.device(self.device))  # Load ckpt for resuming training

    def setup_model(self):
        self.model = self.model.to(self.device)

        if dist.is_initialized():
            if self.sync_bn:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device])

    def setup_optimizer(self):
        if self.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        elif self.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        elif self.optim == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        else:
            raise NotImplementedError

    def save_checkpoint(self, epoch):
        if self.device == 0 or not dist.is_initialized():
            ckpt_dict = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'loss': self.loss_fn,
                         'optimizer_state_dict': self.optimizer.state_dict()}
            torch.save(ckpt_dict, f'{self.ckpt_dir}/ckpt-{(epoch + 1) // self.save_freq}')

            existing_ckpts = os.listdir(self.ckpt_dir)

            if len(existing_ckpts) > self.save_ckpts:
                min_ckpt_idx = min([int(ckpt_path.split('-')[-1]) for ckpt_path in existing_ckpts])
                os.remove(f'{self.ckpt_dir}/ckpt-{min_ckpt_idx}')

    def load_checkpoint(self, map_location=None):
        if self.ckpt_num is None:
            self.ckpt_num = max([int(ckpt_path.split('-')[-1]) for ckpt_path in os.listdir(self.ckpt_dir)])

        ckpt_path = f'{self.ckpt_dir}/ckpt-{self.ckpt_num}'
        assert os.path.exists(ckpt_path), f'checkpoint path {ckpt_path} does NOT exist.'

        ckpt = torch.load(ckpt_path, map_location=map_location)
        ckpt_state_dict = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items() if 'module.' in k}
        model_state_dict = self.model.state_dict()
        ckpt_state_dict_new = {}

        for ckpt_key, model_key in zip(ckpt_state_dict, model_state_dict):
            ckpt_state_dict_new[model_key] = ckpt_state_dict[ckpt_key]  # change ckpt_state_dict keys

        self.model.load_state_dict(ckpt_state_dict_new)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.optimizer.param_groups[0]['capturable'] = True  # For Adam or AdamW
        self.loss_fn = ckpt['loss']
        self.epochs_trained = ckpt['epoch'] + 1

        assert self.epochs_trained == self.ckpt_num * self.save_freq, 'Incorrect trained epochs!'
        self.logger.info(f'Model restored from {ckpt_path}.\n')

    def compute_loss(self, model_out, epoch=None):
        p1, p2, z1, z2 = model_out
        loss = -0.5 * (self.loss_fn(p1, z2).mean() + self.loss_fn(p2, z1).mean())
        reg_loss = self.weight_decay * sum([(para ** 2).sum() for para in self.model.parameters()])

        return loss + reg_loss, loss.view(1,)

    def train_step(self, data1, data2, epoch=None):
        model_out = self.model(data1, data2)
        loss, aux_loss = self.compute_loss(model_out, epoch=epoch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_metrics.update(loss, aux_loss)

    def train_epoch(self, epoch=None):
        self.model.train()

        for data1, data2 in self.train_dataloader:
            data1, data2 = data1.to(self.device), data2.to(self.device)
            self.train_step(data1, data2, epoch=epoch)

    def test_step(self, data1, data2):
        p1, p2, z1, z2 = self.model(data1, data2)
        loss = -0.5 * (self.loss_fn(p1, z2).mean() + self.loss_fn(p2, z1).mean())
        self.test_metrics.update(loss)

    def test_epoch(self):
        self.model.eval()

        with torch.no_grad():
            for data1, data2 in self.test_dataloader:
                data1, data2 = data1.to(self.device), data2.to(self.device)
                self.test_step(data1, data2)

    def eval_epoch(self):
        self.model.eval()

        model = self.model.module if dist.is_initialized() else self.model
        spk_encoder = model.spk_model.spk_encoder

        if hasattr(spk_encoder.emb_layer.fc0, 'act'):
            spk_encoder = torch.nn.Sequential(spk_encoder[:-1], spk_encoder.emb_layer.fc0.linear)

        def extract_emb(eval_dataloader):
            emb = []

            with torch.no_grad():
                for data in eval_dataloader[self.device]:
                    data = data.to(self.device)
                    emb.append(spk_encoder(data)[1]) if model.name.endswith('_ds') else emb.append(spk_encoder(data))

                if dist.is_initialized():
                    emb = dist_concat_all_gather(torch.concat(emb, dim=0))
                else:
                    emb = torch.concat(emb, dim=0)

            return emb.cpu().numpy()

        test_emb = length_norm(extract_emb(self.eval_dataloader['test']))

        if 'vox' in self.train_dataloader.dataset.source:
            test_emb = test_emb[:4874]
            enroll_emb = test_emb.copy()
        else:
            test_emb = test_emb[:17777]
            enroll_emb = length_norm(extract_emb(self.eval_dataloader['enroll']))

        scorer = CosineScorer(enroll_emb, test_emb, self.enroll_ids, self.test_ids, trials_file=self.trials_file)
        scores = scorer.score()
        eer, minDCFs = eval_performance(scores, self.trials_file, [0.01], c_miss=1, c_fa=1)

        return eer, minDCFs[0]

    def train(self):
        self.setup_train()
        self.logger.info(f'No. of total epochs : {self.epochs}, No. of trained epochs : {self.epochs_trained}\n')

        for epoch in range(self.epochs_trained, self.epochs):
            ts = perf_counter()
            self.logger.info(f'Epoch: {epoch}/{self.epochs}')

            # Set learning rate
            for group in self.optimizer.param_groups:
                group['lr'] = self.lr_scheduler.get_lr(epoch)

            self.logger.info(f'lr at epoch {epoch}: {self.optimizer.param_groups[0]["lr"]:.8f}')

            # Train
            self.train_dataloader.batch_sampler.set_epoch(epoch)
            self.train_epoch(epoch)
            train_loss, aux_loss = self.train_metrics.result()

            aux_loss = ', '.join([f'aux_loss{i}: {aux_loss[i]:.3f}' for i in range(len(aux_loss))])
            self.logger.info(f'train_loss: {train_loss:.3f}, {aux_loss}')
            assert not math.isnan(train_loss), 'NaN occurs, training aborted!\n\n\n'
            self.train_metrics.reset()

            # Test
            if self.test_dataloader is not None:
                self.test_epoch()
                test_loss, _ = self.test_metrics.result()

                self.logger.info(f'test_loss: {test_loss:.3f}')
                self.test_metrics.reset()

            # Evaluation
            if self.eval_dataloader is not None:
                eer, min_dcf = self.eval_epoch()
                self.logger.info(f'EER: {eer * 100:.2f}%, minDCF: {min_dcf:.3f}')

            # Save checkpoints
            if (epoch + 1) % self.save_freq == 0:
                self.save_checkpoint(epoch)

            self.logger.info(f'Elapsed time of training epoch {epoch}: {perf_counter() - ts:.2f} s.\n')

        if dist.is_initialized():
            dist.destroy_process_group()

        self.logger.info('[*****] Training finished.\n\n\n')


class LRScheduler(object):
    def __init__(self, lr_conf='lin_1_cos_1:0.1@0,0.01@40,0.001@70'):
        mode, lr_conf = lr_conf.split(':')
        mode = mode.split('_')
        assert len(mode) % 2 == 0, 'Length of mode must be EVEN!'

        self.mode = np.concatenate([[mode[2 * i]] * int(mode[2 * i + 1]) for i in range(len(mode) // 2)])
        self.lrs = [float(lr_.split('@')[0]) for lr_ in lr_conf.split(',')]
        self.milestones = [float(lr_.split('@')[1]) for lr_ in lr_conf.split(',')]
        assert len(self.lrs) == len(self.milestones) == len(self.mode) + 1, 'Misconfig between lrs and modes!'

    @staticmethod
    def linear_lr(start_lr, end_lr, start_epoch, end_epoch, epoch):
        return start_lr + (epoch - start_epoch) * (end_lr - start_lr) / (end_epoch - start_epoch)

    @staticmethod
    def cos_lr(start_lr, end_lr, start_epoch, end_epoch, epoch):
        return start_lr + 0.5 * (end_lr - start_lr) * \
            (1 - math.cos(math.pi * (epoch - start_epoch) / (end_epoch - start_epoch)))

    def get_lr(self, epoch):
        lr = self.lrs[-1]

        for i in range(len(self.mode)):
            if self.milestones[i] <= epoch < self.milestones[i + 1]:
                if self.mode[i] == 'cos':
                    lr = self.cos_lr(self.lrs[i], self.lrs[i + 1], self.milestones[i], self.milestones[i + 1], epoch)
                elif self.mode[i] == 'lin':
                    lr = self.linear_lr(self.lrs[i], self.lrs[i + 1], self.milestones[i], self.milestones[i + 1], epoch)
                else:
                    lr = self.lrs[i]
                break

        return lr


class Metrics(object):
    def __init__(self, device=None):
        self.device = device
        self.loss, self.aux_loss = torch.tensor([0.], device=self.device), None
        self.n_batches = 0

    def reset(self):
        self.loss, self.aux_loss = torch.tensor([0.], device=self.device), None
        self.n_batches = 0

    def update(self, loss, aux_loss=None):
        self.loss += loss
        self.n_batches += 1

        if aux_loss is not None:
            if self.aux_loss is None:
                self.aux_loss = torch.zeros_like(aux_loss)

            self.aux_loss += aux_loss

    def result(self):
        # Get average (loss, aux_loss) per epoch
        if self.n_batches == 0:
            raise ValueError
        else:
            ave_loss = self.loss.item() / self.n_batches

            if self.aux_loss is None:
                ave_aux_loss = (0.,)
            else:
                ave_aux_loss = tuple([loss_.item() / self.n_batches for loss_ in self.aux_loss])

        return ave_loss, ave_aux_loss


def length_norm(data):
    return data / norm(data, axis=1)[:, None] * data.shape[1]  # np.sqrt(data.shape[1])
