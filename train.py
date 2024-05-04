""" Train embedding networks """

import argparse
from datetime import datetime
from front_end.dataset import TrainDataset, TrainBatchSampler, EvalDataset
from front_end.model_moco import MoCo
from front_end.model_moco_ds import MoCoDS
from front_end.trainer_moco import TrainerMoCo
from front_end.trainer_moco_ds import TrainerMoCoDS
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from utils.my_utils import init_logger


# Set global parameters
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--device', default='0,1', help='devices used for training')
parser.add_argument('--port', default='12355', help='port for ddp training')
parser.add_argument('--seed', type=int, default=20230708, help='train dataloader seed')
parser.add_argument('--n_workers', type=int, default=10, help='No. of CPU threads in dataloader')
parser.add_argument('--model', default='moco_ds', help='dsvae, simsiam, simclr, moco_ds, dino')
parser.add_argument('--batch_size', type=int, default=128, help='local mini-batch size')
parser.add_argument('--epochs', type=int, default=50, help='No. of training epochs')
parser.add_argument('--train_src', default='vox2', help='vox2, cnceleb2')
parser.add_argument('--save_ckpts', type=int, default=10, help='No. of ckpts to be saved')
parser.add_argument('--eval', action='store_true', default=True, help='perform validation during training')

parser.add_argument('--feat_dim', type=int, default=80, help='dimension of acoustic features')
parser.add_argument('--min_len', type=int, default=350, help='minimum No. of frames of a training sample')
parser.add_argument('--max_len', type=int, default=350, help='maximum No. of frames of a training sample')
parser.add_argument('--spec_aug', action='store_true', default=True, help='apply SpecAugment')

parser.add_argument('--filters', default='512-512-512-512-1536',
                    help='No. of channels of convolutional layers, 512-512-512-512-512-512-512-512-1536')
parser.add_argument('--kernel_sizes', default='5-3-3-3-1',
                    help='kernel size of convolutional layers, 5-3-3-1-1, 5-1-3-1-3-1-3-1-1')
parser.add_argument('--dilations', default='1-2-3-4-1',
                    help='dilation of convolutional layers, 1-2-3-1-1, 1-1-2-1-3-1-4-1-1')
parser.add_argument('--pooling', default='ctdstats-128-1', help='stats, attention-256-1, ctdstats-256-0')
parser.add_argument('--embedding_dims', default='192', help='embedding network config, 512-512')
parser.add_argument('--moco_cfg', default='65536-0.999-0.05', help='len_queue-momentum-temperature')
parser.add_argument('--optim', default='sgd', help='adam or sgd')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay')
parser.add_argument('--sync_bn', action='store_true', default=True, help='apply SyncBatchNorm')
parser.add_argument('--lr', default='lin_1_cos_1:0.01@0,0.05@15,0.002@50')
parser.add_argument('--ckpt_dir', nargs='?', help='directory of model checkpoint')
parser.add_argument('--ckpt_num', nargs='?', type=int, help='checkpoint number for resuming training, default: None')
parser.add_argument('--save_freq', type=int, default=1, help='frequency to save the model')
parser.add_argument('--log_dir', default='log', help='log directory')
args = parser.parse_args()


def train_func(rank, n_gpus):
    # --------------------------------------------------------------------------------------------------
    # Initialize ckpt_dir and logger
    # --------------------------------------------------------------------------------------------------
    cur_time = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = f'model_ckpt/ckpt_{cur_time}' if args.ckpt_dir is None else args.ckpt_dir
    ckpt_time = '_'.join(ckpt_dir.split('/')[-1].split('_')[1:])
    Path(f'{ckpt_dir}').mkdir(parents=True, exist_ok=True)

    Path(f'{args.log_dir}').mkdir(parents=True, exist_ok=True)
    log_path = f'{args.log_dir}/log_{ckpt_time}.log'
    logger = init_logger(logger_name='train', log_path=log_path, device=rank, n_gpus=n_gpus)

    logger.info('----------------------------------------------------')
    logger.info(f'[*] ckpt_dir: {ckpt_dir}')

    for arg, val in vars(args).items():
        if arg not in ['ckpt_dir']:
            logger.info(f'[*] {arg}: {val}')
    logger.info('----------------------------------------------------\n')

    # --------------------------------------------------------------------------------------------------
    # Initialize DDP
    # --------------------------------------------------------------------------------------------------
    if n_gpus > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', rank=rank, world_size=n_gpus)

    # --------------------------------------------------------------------------------------------------
    # Create training and test dataloaders
    # --------------------------------------------------------------------------------------------------
    train_dataset = TrainDataset(
        source=args.train_src, mode='train', min_len=args.min_len, max_len=args.max_len)
    train_sampler = TrainBatchSampler(train_dataset, batch_size=args.batch_size, seed=args.seed)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_sampler=train_sampler, num_workers=args.n_workers,
        collate_fn=train_dataset.segment_batch)

    # test_dataset = TrainDataset(
    #     mode='val', min_len=args.min_len, max_len=args.max_len, batch_size=args.batch_size,
    #     repeat_rate=1.)
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=None, num_workers=args.n_workers)

    if args.eval:
        eval_dataloader = {'enroll': [], 'test': []}
        src = 'voxceleb1' if 'vox' in args.train_src else 'cnceleb1'
        keys = ['test'] if 'vox' in args.train_src else ['enroll', 'test']

        for k in keys:
            n_utts = EvalDataset(source=f'{src}_{k}').n_utterance

            for i in range(n_gpus):
                start_idx, end_idx = i * (n_utts // n_gpus), (i + 1) * (n_utts // n_gpus)
                dataset = EvalDataset(source=f'{src}_{k}', start=start_idx, end=end_idx)
                eval_dataloader[k].append(DataLoader(dataset=dataset, num_workers=2))
    else:
        eval_dataloader = None

    # --------------------------------------------------------------------------------------------------
    # Create model
    # --------------------------------------------------------------------------------------------------
    model_args = {
        'feat_dim': args.feat_dim, 'filters': args.filters, 'kernel_sizes': args.kernel_sizes,
        'dilations': args.dilations, 'pooling': args.pooling, 'embedding_dims': args.embedding_dims,
        'spec_aug': args.spec_aug, 'name': args.model}

    if args.model == 'moco':
        model = MoCo(**model_args, moco_cfg=args.moco_cfg)
    elif args.model == 'moco_ds':
        model = MoCoDS(
            **model_args, moco_cfg=args.moco_cfg, n_shared_layers=4, content_hidden_dim=512, content_emb_dim=16,
            dec_filters='512-80', dec_kernel_sizes='3-5', dec_dilations='2-1')
    else:
        raise NotImplementedError

    logger.info('===============================================')
    logger.info(model)
    total_paras = sum(para.numel() for para in model.parameters() if para.requires_grad)
    logger.info(f'Total No. of parameters: {total_paras / 1e6:.3f} M\n')
    logger.info('===============================================\n')

    # --------------------------------------------------------------------------------------------------
    # Create trainer
    # --------------------------------------------------------------------------------------------------
    trainer_args = {
        'train_dataloader': train_dataloader, 'test_dataloader': None, 'eval_dataloader': eval_dataloader,
        'model': model, 'optim': args.optim, 'weight_decay': args.weight_decay, 'lr': args.lr,
        'epochs': args.epochs, 'device': rank, 'sync_bn': args.sync_bn, 'ckpt_dir': ckpt_dir,
        'ckpt_num': args.ckpt_num, 'save_freq': args.save_freq, 'logger': logger}

    if args.model == 'moco':
        trainer = TrainerMoCo(**trainer_args)
    elif args.model == 'moco_ds':
        trainer = TrainerMoCoDS(**trainer_args)
    else:
        raise NotImplementedError

    return trainer.train()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    num_gpus = len(args.device.split(','))

    if num_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        torch.multiprocessing.spawn(train_func, args=(num_gpus, ), nprocs=num_gpus, join=True)
    else:
        train_func(int(args.device), 1)

    print('To the END.\n\n')
