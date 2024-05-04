""" Backend processing for Voxceleb1 """

import numpy as np
from time import perf_counter
import argparse
from utils.my_utils import init_logger, rd_data_frame
from front_end.trainer import length_norm
from back_end.cosine_score import CosineScorer
from utils.eval_metrics import eval_performance


# Set parameters
parser = argparse.ArgumentParser(description='be_voxceleb')
parser.add_argument('--eval_dir', default='./eval', help='directory of evaluation')
parser.add_argument('--eval_meta_dir', default='./meta/eval', help='directory of evaluation meta info')
parser.add_argument('--trials_dir', default='./trials', help='directory of evaluation meta info')
parser.add_argument('--scoring_type', default='multi_session', help='multi_session, ivec_averaging, score_averaging')
parser.add_argument('--is_snorm', action='store_true', default=False, help='perform S-norm')
parser.add_argument('--n_top', type=int, default=150, help='ratio of top cohort scores selected for snorm')
parser.add_argument('--ckpt_time', default='20230601_0946', help='checkpoint time')
parser.add_argument('--ckpt_num', type=int, default=131, help='index. of checkpoint')
args = parser.parse_args()

logger = init_logger(logger_name='voxceleb_be', log_path='log/be_voxceleb.log')
# logger.info(f'-------------------------------------')
# for arg, val in vars(args).items():
#     logger.info(f'[*] {arg}: {val}')
# logger.info(f'-------------------------------------')


# ------------------------------------------------------------------------------------------------
# Load cohort for snorm
# ------------------------------------------------------------------------------------------------
X_snorm = None

if args.is_snorm:
    trn_xvec_file = f'{args.eval_dir}/xvectors/plda_vox2_train/xvector_{args.ckpt_time}_{args.ckpt_num}.npy'
    X_snorm = np.load(trn_xvec_file)
    X_snorm -= X_snorm.mean(0)
    X_snorm = length_norm(X_snorm)


# ------------------------------------------------------------------------------------------------
# Compute scores
# ------------------------------------------------------------------------------------------------
# Load x-vectors
enroll_xvecs_file = f'{args.eval_dir}/xvectors/voxceleb1_test/xvector_{args.ckpt_time}_{args.ckpt_num}.npy'
enroll_info_file = f'{args.eval_meta_dir}/voxceleb1_test_path2info'

X_enroll = np.load(enroll_xvecs_file)
enroll_ids = rd_data_frame(enroll_info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])['utt_id'].values

X_enroll = length_norm(X_enroll)
X_test = X_enroll.copy()
test_ids = enroll_ids.copy()

# Perform scoring
trials_file = f'{args.trials_dir}/voxceleb1/trials_voxceleb'
scores_file = f'{args.eval_dir}/scores/scores_voxceleb1.npz'
scores_snorm_file = f'{args.eval_dir}/scores/scores_voxceleb1_snorm.npz'

print(f'voxceleb1_test Cosine scoring...')
t_s = perf_counter()
scorer = CosineScorer(
    X_enroll, X_test, enroll_ids, test_ids, trials_file=trials_file, scoring_type=args.scoring_type)
scorer.score(scores_file)
print(f'Time of the voxceleb1 Cosine scoring: {perf_counter() - t_s}s.\n')

if args.is_snorm:
    print(f'voxceleb1 Cosine scoring with S-norm...')
    t_s = perf_counter()
    scorer.score(scores_snorm_file, X_snorm, X_snorm, is_snorm=True, n_top=args.n_top)
    print(f'Time of the voxceleb1 Cosine scoring with S-norm: {perf_counter() - t_s}s.\n')


# ------------------------------------------------------------------------------------------------
# Compute EER, minDCF and actDCF
# ------------------------------------------------------------------------------------------------
trials_file = f'{trials_file}.npz'
p_targets = [0.01, 0.001]

logger.info(f'==============================================================================')
logger.info(f'voxceleb1_test, ckpt_time: {args.ckpt_time}, ckpt_num: {args.ckpt_num}...')
eer, minDCFs = eval_performance(scores_file, trials_file, p_targets, c_miss=1, c_fa=1)
# logger.info(f'EER: {eer * 100:.2f} %, minDCF (p_tar=1e-2): {minDCFs[0]:.3f}, minDCF (p_tar=1e-3): {minDCFs[1]:.3f}')
logger.info(f'EER: {eer * 100:.2f} %, minDCF (p_tar=1e-2): {minDCFs[0]:.3f}')
logger.info(f'==============================================================================\n')

if args.is_snorm:
    logger.info(f'==============================================================================')
    logger.info(f'voxceleb1_test-snorm, ckpt_time: {args.ckpt_time}, ckpt_num: {args.ckpt_num}...')
    eer, minDCFs = eval_performance(scores_snorm_file, trials_file, p_targets, c_miss=1, c_fa=1)
    logger.info(f'EER: {eer * 100:.2f} %, minDCF (p_tar=1e-2): {minDCFs[0]:.3f}')
    logger.info(f'==============================================================================\n')

logger.info('To the END.\n\n')
