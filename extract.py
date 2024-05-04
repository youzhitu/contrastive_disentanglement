""" Extract embeddings """

import argparse
from front_end.model_moco import MoCo
from front_end.model_moco_ds import MoCoDS
from front_end.extractor import Extractor
import numpy as np
import os
import shutil
from time import perf_counter
from utils.my_utils import select_plda_trn_info


# Set global parameters
parser = argparse.ArgumentParser(description='extract')
parser.add_argument('--model', default='moco', help='simsiam, simclr, moco, dino')
parser.add_argument('--filters', default='512-512-512-512-1536',
                    help='No. of channels of convolutional layers, 512-512-512-512-512-512-512-512-1536')
parser.add_argument('--kernel_sizes', default='5-3-3-3-1',
                    help='kernel size of convolutional layers, 5-1-3-1-3-1-3-1-1')
parser.add_argument('--dilations', default='1-2-3-4-1', help='dilation of convolutional layers, 1-1-2-1-3-1-4-1-1')
parser.add_argument('--pooling', default='ctdstats-128-1', help='stats, attention-500-1, ctdstats-256-0')
parser.add_argument('--embedding_dims', default='192', help='embedding network config, 512-512')
parser.add_argument('--predictor_dims', default='2048-2048-128', help='prediction network config')
parser.add_argument('--moco_cfg', default='65536-0.999-0.05', help='len_queue-momentum-temperature')

parser.add_argument('--feat_dim', type=int, default=80, help='dimension of acoustic features')
parser.add_argument('--n_workers', type=int, default=2, help='No. of workers used in the dataloader')
parser.add_argument('--device', default='0', help='cuda, cpu')
parser.add_argument('--ckpt_dir', nargs='?', help='directory of model checkpoint')
parser.add_argument('--ckpt_num', nargs='?', type=int, help='checkpoint number for resuming training, default: None')
parser.add_argument('--n_jobs', type=int, default=1, help='No. of jobs for extracting x-vectors')
parser.add_argument('--job', type=int, default=0, help='job index of extraction')
parser.add_argument('--selected_dur', nargs='?', type=int,
                    help='duration of randomly selected utterances in No. of frames, e.g., 500 means 5s.')

parser.add_argument('--eval_dir', default='eval', help='directory of evaluation')
parser.add_argument('--task', default='voxceleb1', help='voxceleb1, voices19, sre, cnceleb1, ffsvc')
args = parser.parse_args()

print(f'-------------------------------------')
for arg, val in vars(args).items():
    print(f'[*] {arg}: {val}')
print(f'-------------------------------------\n')


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------
    # Create a model instance
    # ------------------------------------------------------------------------------------------------
    model_args = {
        'feat_dim': args.feat_dim, 'filters': args.filters, 'kernel_sizes': args.kernel_sizes,
        'dilations': args.dilations, 'pooling': args.pooling, 'embedding_dims': args.embedding_dims,
        'predictor_dims': args.predictor_dims}

    if args.model == 'moco':
        model = MoCo(**model_args, moco_cfg=args.moco_cfg)
    elif args.model == 'moco_ds':
        model = MoCoDS(**model_args, moco_cfg=args.moco_cfg)
    else:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------
    # Set data sources for x-vector extraction and select info for PLDA training
    # ------------------------------------------------------------------------------------------------
    meta_dir = 'meta/eval'
    plda_dir = f'{args.eval_dir}/plda'
    os.makedirs(plda_dir, exist_ok=True)

    print('Setting data sources...')
    ts = perf_counter()

    if args.task == 'voxceleb1':
        # if args.job == 0:
        #     select_plda_trn_info(meta_dir, plda_dir, source='vox2_train', n_utts_per_spk=1)

        # data_sources = ['plda_vox2_train', 'voxceleb1_test']
        data_sources = ['voxceleb1_test']
        n_utts_per_partition = 10000
    elif args.task == 'cnceleb1':
        # if args.job == 10:
        #     select_plda_trn_info(meta_dir, plda_dir, source='cnceleb1_train')

        data_sources = ['cnceleb1_train', 'cnceleb1_enroll', 'cnceleb1_test']
        n_utts_per_partition = 2000
    else:
        raise NotImplementedError

    print(f'time of subset selection: {perf_counter() - ts} s\n')

    # ------------------------------------------------------------------------------------------------
    # Extract x-vectors
    # ------------------------------------------------------------------------------------------------
    for source in data_sources:
        print(f'Extracting x-vectors for {source}...')

        ts = perf_counter()

        ckpt_time = '_'.join(args.ckpt_dir.split('/')[-1].split('_')[-2:])
        if args.ckpt_num is None:
            ckpt_num = max([int(ckpt.split('-')[-1]) for ckpt in os.listdir(args.ckpt_dir)])
        else:
            ckpt_num = args.ckpt_num
        ckpt_path = f'{args.ckpt_dir}/ckpt-{ckpt_num}'

        extract_dir = f'extract/{ckpt_time}_{ckpt_num}_{source}'  # for saving temporary x-vectors
        os.makedirs(extract_dir, exist_ok=True)
        xvec_dir = f'{args.eval_dir}/xvectors/{source}'  # for saving final x-vectors
        os.makedirs(xvec_dir, exist_ok=True)

        extractor = Extractor(source=source, model=model, ckpt_path=ckpt_path, device=args.device,
                              extract_dir=extract_dir, n_utts_per_partition=n_utts_per_partition,
                              n_jobs=args.n_jobs, job=args.job, selected_dur=args.selected_dur,
                              n_workers=args.n_workers)
        xvectors = extractor.extract()

        if xvectors.size:
            np.save(f'{xvec_dir}/xvector_{ckpt_time}_{ckpt_num}.npy', xvectors)
            shutil.rmtree(extract_dir)  # Remove temporary x-vectors

        print(f'time of extraction: {perf_counter() - ts} s\n')

    print('To the END.\n\n')
    print()
