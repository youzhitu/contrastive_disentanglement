""" Cosine scoring """

import numpy as np
import pandas as pd
from pathlib import Path
from utils.my_utils import get_filtered_idx


class CosineScorer(object):
    def __init__(self, X_enroll, X_test, enroll_ids, test_ids, trials_file='trials', sgplda_paras_file='sgplda.npz',
                 scoring_type='multi_session'):
        """ scoring_type: 'multi_session', 'ivec_averaging' or 'score_averaging' """

        self.X_enroll = X_enroll
        self.X_test = X_test
        self.enroll_ids = enroll_ids
        self.test_ids = test_ids
        self.trials_file = trials_file
        self.sgplda_paras_file = sgplda_paras_file
        self.scoring_type = scoring_type

        self._init_scorer()

    def _init_scorer(self):
        self._load_trials()
        self._load_trials_idx()

    def _load_trials(self):
        # print('Loading trials ...')
        self.trials_enroll_ids, self.trials_test_ids = load_trials(self.trials_file)

    def _load_trials_idx(self):
        # print('Indexing trials ...')
        self.X_enroll_idx, self.n_utts_per_spk, self.X_test_idx = \
            index_trials(self.trials_file, self.enroll_ids, self.test_ids, self.trials_enroll_ids, self.trials_test_ids)

    def score(self, scores_file=None, X_enroll_in=None, X_test_in=None, is_snorm=False, select_ratio=0.1, n_top=500):
        """ Scoring main function
        If X_enroll_in and X_test_in are not provided, which is the default setting, scores will be computed
        according to the trials list.
        If both X_enroll_in and X_test_in are provided, but is_snorm is disabled, scores will be simply
        computed for each {X_enroll_in, X_test_in} pair.
        If both X_enroll_in and X_test_in are provided, and is_snorm is enabled, Snorm will be performed in which
        snorm scores ("scores_ec" and "scores_ct") are first computed and raw scores are then normalized.

        Snorm consists of two stages: Znorm and Tnorm. Note that if multi-session scoring is selected,
        the "is_tnorm" flag should be enabled to make sure the "enroll_ivecs" in the scoring function
        are correctly picked up. """

        if X_enroll_in is not None and X_test_in is not None:
            if is_snorm:
                scores_ec = (self.X_enroll @ X_enroll_in.T).ravel()
                scores_ct = (X_test_in @ self.X_test.T).ravel()
                scores = self.cosine_scoring(self.X_enroll, self.X_test, is_trials_scoring=True, is_tnorm=False)
                scores = self.snorm(scores_ec, scores_ct, scores, select_ratio=select_ratio, n_top=n_top)
            else:
                scores = self.cosine_scoring(X_enroll_in, X_test_in, is_trials_scoring=False, is_tnorm=True)
        else:
            scores = self.cosine_scoring(self.X_enroll, self.X_test, is_trials_scoring=True, is_tnorm=False)

        if scores_file is not None:
            self.save_scores(scores_file, scores)

        return scores

    def cosine_scoring(self, X_enroll, X_test, is_trials_scoring=True, is_tnorm=False):
        """ Compute scores for every enrolled speaker
            :param
            is_trials_scoring: indicating whether the current scoring is trials scoring or S-norm scoring
                default: trials scoring
            is_tnorm: indicating whether the current S-norm scoring is Z-norm scoring or T-norm scoring
                Note that it is only valid when is_trials_scoring is disabled, default: False (Z-norm scoring)
            """

        n_unique_enroll_ids = X_enroll.shape[0] if is_tnorm else self.n_utts_per_spk.shape[0]
        scores = []

        for i in range(n_unique_enroll_ids):
            enroll_ivecs = X_enroll[self.X_enroll_idx[i]] if not is_tnorm else np.expand_dims(X_enroll[i], axis=0)
            test_ivecs = X_test[self.X_test_idx[i]] if is_trials_scoring else X_test

            if self.scoring_type == 'score_averaging':
                cos_scores = np.einsum('ik, jk->ij', enroll_ivecs, test_ivecs)
                cos_scores = cos_scores.reshape(enroll_ivecs.shape[0], -1).mean(0)
            else:
                cos_scores = test_ivecs @ enroll_ivecs.mean(0)

            scores.append(cos_scores)

        return np.concatenate(scores)

    def save_scores(self, scores_file, scores):
        Path(Path(scores_file).parent).mkdir(exist_ok=True)

        if scores_file.endswith('npz'):
            np.savez(scores_file, enroll=self.trials_enroll_ids, test=self.trials_test_ids, llr=scores)
        else:
            scores = np.vstack([self.trials_enroll_ids, self.trials_test_ids, scores]).T
            pd.DataFrame(scores).to_csv(scores_file, index=False, header=False, sep=' ')

    def snorm(self, scores_ec, scores_ct, scores, select_ratio=0.1, n_top=500):
        """ Adaptive S-norm of scores
            select_ratio: ratio of the top scores selected for each enrollment or test ivector for S-norm
            Note that the order of the T-norm scoring pair (cohort v.s. test or test v.s. cohort) does not affect the
            speed of Snorm. """

        N_ec_se = int(scores_ec.shape[0] / self.n_utts_per_spk.shape[0] * select_ratio) if n_top is None else n_top
        N_ct_se = int(scores_ct.shape[0] / self.X_test.shape[0] * select_ratio) if n_top is None else n_top

        scores_ec = np.sort(scores_ec.reshape(self.n_utts_per_spk.shape[0], -1), axis=1, kind='mergesort')[:, -N_ec_se:]
        mu_ec, std_ec = np.mean(scores_ec, axis=1), np.std(scores_ec, axis=1)

        scores_ct = np.sort(scores_ct.reshape(-1, self.X_test.shape[0]), axis=0, kind='mergesort')[-N_ct_se:]
        mu_tc, std_tc = np.mean(scores_ct, axis=0), np.std(scores_ct, axis=0)

        trials_enroll_ids_idx = get_filtered_idx(self.enroll_ids, self.trials_enroll_ids)
        mu_ec, std_ec = mu_ec[trials_enroll_ids_idx], std_ec[trials_enroll_ids_idx]

        trials_test_ids_idx = get_filtered_idx(self.test_ids, self.trials_test_ids)
        mu_tc, std_tc = mu_tc[trials_test_ids_idx], std_tc[trials_test_ids_idx]

        return ((scores - mu_ec) / std_ec + (scores - mu_tc) / std_tc) / 2


def load_trials(trials_file):
    """
    Load trials list
    The trials list should be sorted by 'enroll_id' and can be different with the original trials. This is
    for accelerating scoring, which requires an indexing operation on the eval data (see index_trials()).
    :return
        trials_enroll_ids, trials_test_ids: np tensor, [len(trials), ]
    """

    trials_file = f'{trials_file}.npz' if Path(f'{trials_file}.npz').exists() else trials_file

    if trials_file.endswith('.npz'):
        trails = np.load(trials_file, allow_pickle=True)
        return trails['enroll'], trails['test']
    else:
        trials = pd.read_csv(trials_file, sep=' ', names=['enroll', 'test', 'key'], dtype=str)
        trials = trials.sort_values(by='enroll', kind='mergesort')
        trials_enroll_ids, trials_test_ids = trials['enroll'].values, trials['test'].values
        np.savez(f'{trials_file}.npz', enroll=trials_enroll_ids, test=trials_test_ids, key=trials['key'].values)

        return trials_enroll_ids, trials_test_ids


def index_trials(trials_file, enroll_ids=None, test_ids=None, trials_enroll_ids=None, trials_test_ids=None):
    """
    Index enrollment and test data for accelerating scoring
    Note the enrolled speaker IDs in the trials list (trials_enroll_ids) must be sorted.
    :param trials_file: str
    :param enroll_ids: ndarray, speaker IDs of enrollment data
    :param test_ids: ndarray, utterances IDs of test data
    :param trials_enroll_ids: ndarray, enrolled speaker IDs in the trials list
    :param trials_test_ids: ndarray, test utterances IDs in the trials list
    :return:
        enroll_idx: list, index to select unique enrolled speakers in the enroll_ids,
                            len(enroll_idx) == len(enroll_ids_unique)
        n_utts_per_spk_enroll: ndarray, No. of utterances for each enrolled speaker
        test_idx: list, index to select test utterances that corresponds to each unique
                            enrolled speaker in the trials pairs, len(test_idx) == len(enroll_ids_unique)
    """

    file_split = trials_file.split('/')
    trial_name = file_split[-1].split('.')[0]
    index_file = f'{"/".join(file_split[:-1])}/index_{trial_name}.npz'

    if Path(index_file).exists():
        eval_index_npz = np.load(index_file, allow_pickle=True)
        enroll_idx, n_utts_per_spk_enroll, test_idx = eval_index_npz['enroll'], eval_index_npz['n_utt'], \
            eval_index_npz['test']
        enroll_idx = [x.astype(int) for x in enroll_idx]
        test_idx = [x.astype(int) for x in test_idx]

        return enroll_idx, n_utts_per_spk_enroll, test_idx
    else:
        enroll_ids_unique, n_utts_per_spk_enroll = np.unique(enroll_ids, return_counts=True)
        enroll_idx, test_idx = [], []

        for enroll_id in enroll_ids_unique:
            enroll_id_se_idx = np.where(enroll_ids == enroll_id)[0]
            trials_test_id_se = trials_test_ids[trials_enroll_ids == enroll_id]
            test_id_se_idx = get_filtered_idx(test_ids, trials_test_id_se)

            enroll_idx.append(enroll_id_se_idx)
            test_idx.append(test_id_se_idx)

        np.savez(index_file, enroll=np.asarray(enroll_idx, dtype='object'), n_utt=n_utts_per_spk_enroll,
                 test=np.asarray(test_idx, dtype='object'))

        return enroll_idx, n_utts_per_spk_enroll, test_idx
