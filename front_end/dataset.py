""" Dataset """

import numpy as np
from utils.my_utils import rd_data_frame
import random
from torch.utils.data import Dataset, Sampler
import torch
import torchaudio
import torch.distributed as dist
from scipy import signal


cwd = '.'


class TrainDataset(Dataset):
    def __init__(self, source='vox2', mode='train', min_len=300, max_len=300):
        super().__init__()
        """
        :param mode: str, 'train' or 'val'
        :param min_len: int, minimum No. of frames of a training sample
        :param max_len: int, maximum No. of frames of a training sample
        """

        self.source = source
        self.mode = mode
        self.min_len = min_len
        self.max_len = max_len

        self.sample_len = max_len * 160 + 240  # 1 frame = 10ms@16kHz
        self._load_wav_info()
        self.noise_reverb_augmentor = NoiseReverbAugment()

    def _load_wav_info(self):
        path2info_file = f'{cwd}/meta/{self.source}/{self.mode}_path2info'
        self.path2info = rd_data_frame(path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur', 'label'])
        self.path2info = self.path2info[['utt_path', 'n_sample']]
        self.n_utterance = self.path2info.shape[0]
        print(f'No. of utterances: {self.n_utterance}\n')

    def __len__(self):
        return self.n_utterance

    def __getitem__(self, idx):
        utt_path, utt_len = self.path2info.iloc[idx]
        wavs = load_wave(utt_path, utt_len, self.sample_len, n_wavs=2)
        wavs1 = [self.noise_reverb_augmentor.augment(wavs[i]) for i in range(2)]

        return wavs1[0].astype(np.float32), wavs1[1].astype(np.float32)

    def segment_batch(self, batch_data):
        """ collate_fn """

        if self.min_len == self.max_len:
            data, data1 = list(zip(*batch_data))

            return torch.tensor(np.stack(data)), torch.tensor(np.stack(data1))

        seg_len = random.randint(self.min_len, self.sample_len) * 160 + 240
        seg_batch_data, seg_batch_data1 = [], []

        for data, data1 in batch_data:
            seg_offset = random.randint(0, self.sample_len - seg_len)
            seg_batch_data.append(data[seg_offset: seg_offset + seg_len])
            seg_batch_data1.append(data1[seg_offset: seg_offset + seg_len])

        return torch.from_numpy(np.asarray(seg_batch_data)), torch.from_numpy(np.asarray(seg_batch_data1))


class TrainBatchSampler(Sampler):
    def __init__(self, data_source, batch_size=128, seed=12345):
        super().__init__(data_source)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        self.n_utterance = data_source.n_utterance
        self._compute_num_batches()

    def _compute_num_batches(self):
        self.batches = self.n_utterance // self.batch_size

        if dist.is_initialized():
            self.batches //= dist.get_world_size()

    def __len__(self):
        return self.batches

    def __iter__(self):
        per_rank_index = self.sample_per_rank_index(self.n_utterance)
        per_rank_index = np.stack([per_rank_index[i * self.batch_size: (i + 1) * self.batch_size]
                                   for i in range(self.batches)], axis=0)

        return iter(per_rank_index)

    def sample_per_rank_index(self, n_paths, iteration=0):
        rng = np.random.default_rng(seed=self.seed + self.epoch + iteration)
        per_rank_index = rng.permutation(n_paths)

        if dist.is_initialized():
            rank = dist.get_rank()
            n_paths //= dist.get_world_size()
            per_rank_index = per_rank_index[rank * n_paths: (rank + 1) * n_paths]

        return per_rank_index

    def set_epoch(self, epoch):
        self.epoch = epoch


class EvalDataset(Dataset):
    def __init__(self, source='voxceleb1_test', feat_dim=40, start=None, end=None, selected_dur=None):
        super().__init__()
        """
        Dataset of an evaluation data partition, partition = whole_data[start: end]
        :param start: start index of the data partition
        :param end: end index of the partition
        :param selected_dur: duration of randomly selected segments in No. of frames
                This is for duration mismatch experiments. 'None' means using full-length utts.
        """
        self.source = source
        self.feat_dim = feat_dim
        self.start = start
        self.end = end
        self.selected_dur = selected_dur

        self._load_feat_info()

    def _load_feat_info(self):
        path2info_file = f'{cwd}/meta/eval/{self.source}_path2info'
        self.path2info = rd_data_frame(path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])
        self.path2info = self.path2info[['utt_path', 'n_sample']]
        self.n_utterance = self.path2info.shape[0]

        self.start = 0 if self.start is None else self.start
        self.end = self.path2info.shape[0] if self.end is None else self.end
        print(f'length of partition: {self.end - self.start}\n')

    def __len__(self):
        return max(self.end - self.start, 0)

    def __getitem__(self, idx):
        idx += self.start
        path, n_sample = self.path2info.iloc[idx].values
        sample_len = self.selected_dur * 160 + 240 if self.selected_dur is not None else n_sample
        wav = load_wave(path, n_sample, sample_len)[0]

        return wav


class NoiseReverbAugment(object):
    """ Add noise (MUSAN) and perform reverberation (RIR)
     MUSAN split refers to https://github.com/clovaai/voxceleb_trainer/blob/master/dataprep.py """
    def __init__(self):
        self.aug_srcs = ['speech', 'music', 'noise', 'rir']
        self.musan_snrs = {'speech': [13, 20], 'noise': [0, 15], 'music': [5, 15]}
        self.n_musan_augs = {'speech': [1, 1], 'noise': [1, 1], 'music': [1, 1]}
        self.aug_path2info = {}

        for src in self.aug_srcs:
            path2info_file = f'{cwd}/meta/musan_rir/{src}_path2dur'
            path2info = rd_data_frame(path2info_file, ['utt_path', 'n_sample', 'dur'])
            self.aug_path2info[src] = path2info[['utt_path', 'n_sample']]

    def augment(self, wav):
        """
        :param wav: np tensor, [1, sample_length]
        :return: wav: np tensor, [1, sample_length]
        """

        speech = self.make_musan(wav, 'speech')
        music = self.make_musan(wav, 'music')
        noise = self.make_musan(wav, 'noise')

        wav = self.reverberate(wav)
        wav += (speech + music + noise)

        return wav

    def make_musan(self, wav, musan_src):
        """
        :param wav: np tensor, [sample_length,]
        :param musan_src: str, 'music', 'speech', or 'noise'
        :return:
            noisy wav: np tensor, [sample_length,]
        """

        wav_len = wav.shape[0]
        # p_wav = np.sum(wav ** 2) + 1e-5
        p_wav = 10 * np.log10(np.sum(wav ** 2) + 1e-5)

        n_augs = random.randint(self.n_musan_augs[musan_src][0], self.n_musan_augs[musan_src][1])
        path2info = self.aug_path2info[musan_src].sample(n_augs)
        snrs = np.random.uniform(self.musan_snrs[musan_src][0], self.musan_snrs[musan_src][1], n_augs)

        noise = np.stack([load_wave(path, n_sample, wav_len)[0] for path, n_sample in path2info.values], axis=0)
        # p_noise = np.sum(noise ** 2, axis=1) + 1e-5
        # noise *= (np.sqrt(p_wav / p_noise) * 10 ** (-snrs / 20))[:, None]
        p_noise = 10 * np.log10(np.sum(noise ** 2, axis=1) + 1e-5)
        noise *= np.sqrt(10 ** ((p_wav - p_noise - snrs) / 10))[:, None]

        return np.sum(noise, axis=0)  # + wav

    def reverberate(self, wav):
        """
        :param wav: np tensor, [sample_length,]
        :return:
            reverberated wav: np tensor, [sample_length,]
        """

        wav_len = wav.shape[0]
        path, n_sample = self.aug_path2info['rir'].sample(1).values[0]
        rir = load_wave(path, n_sample, n_sample)[0]
        rir = rir / np.sqrt(np.sum(rir ** 2) + 1e-5)

        return signal.convolve(wav, rir, mode='full')[:wav_len]


def load_wave(wav_file, wav_length, sample_length, n_wavs=1):
    """
    :param wav_file: str
    :param wav_length: int, number of sample points of a wave
    :param sample_length: int, length to be sampled
    :param n_wavs: int, No. of wave segments to load
    :return:
        wav: list of np tensor, the length of list is n_wavs and each element is of shape [sample_length,]
    """

    # noinspection PyUnresolvedReferences
    wav = torchaudio.load(wav_file)[0][0].numpy()
    wavs_out = []

    if wav_length > sample_length:
        if n_wavs > 1:
            start = np.random.randint(0, wav_length - sample_length, n_wavs)
        else:
            start = [np.random.randint(0, wav_length - sample_length)]

        for s in start:
            wavs_out.append(wav[s: s + sample_length])
    else:
        for _ in range(n_wavs):
            wavs_out.append(np.pad(wav, (0, sample_length - wav_length), 'wrap'))

    return wavs_out
