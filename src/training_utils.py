import torch
import numpy as np
import os.path as osp
from audio_augmentations import AudioTransform,Compose,OneOf,GaussianNoiseSNR,PinkNoiseSNR,TimeShift,VolumeControl
from spectrogram_utils import mono_to_color,normalize
import torchaudio
import pandas as pd
import librosa
import albumentations as A
import random

def LWLRAP(preds, labels):
    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes).to(preds.device)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(
        ground_truth_ranks, dim=-1, descending=False)
    # Number of GT labels per instance
    num_labels = labels.sum(-1)
    pos_matrix = torch.tensor(
        np.array([i+1 for i in range(labels.size(-1))])).unsqueeze(0).to(preds.device)
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = scores.sum() / labels.sum()
    return score.item()


class RFCDataset:
    def __init__(self, tp, fp=None, config=None,
                 mode='train', inv_counts=None):
        self.tp = tp
        self.fp = pd.read_csv("../input/rfcxextras/cornell-train.csv")
        self.fp = self.fp[self.fp.ebird_code<'c'].reset_index(drop=True)
        self.fp_root = "../input/birdsong-resampled-train-audio-00/"        
        self.inv_counts = inv_counts
        self.config = config
        self.sr = self.config.sr
        self.total_duration = self.config.total_duration
        self.duration = self.config.duration
        self.data_root = self.config.TRAIN_AUDIO_ROOT
        self.nmels = self.config.nmels
        self.fmin, self.fmax = 84, self.sr//2
        self.mode = mode
        self.num_classes = self.config.num_classes
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=48_000, new_freq=self.sr)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_mels=self.nmels,
                                                        f_min=self.fmin, f_max=self.fmax,
                                                        n_fft=2048)
        self.transform = Compose([
            OneOf([
                GaussianNoiseSNR(min_snr=10),
                PinkNoiseSNR(min_snr=10)
            ]),
            TimeShift(sr=self.sr),
            VolumeControl(p=0.5)
        ])
        self.img_transform = A.Compose([
            A.OneOf([
                A.Cutout(max_h_size=5, max_w_size=20),
                A.CoarseDropout(max_holes=4),
                A.RandomBrightness(p=0.25),
            ], p=0.5)])
        self.num_splits = self.config.total_duration//self.duration
        assert self.config.total_duration == self.duration * \
            self.num_splits, "not a multiple"

    def __len__(self):
        return len(self.tp)

    def __getitem__(self, idx):
        labels = np.zeros((self.num_classes,), dtype=np.float32) # Number of artists here

        song_id = self.tp.loc[idx, 'song_id'] ## Get the song ID here
        df = self.tp.loc[self.tp.song_id == song_id]
        maybe_labels = df.artist.unique()
        np.put(labels, maybe_labels, 0.2)

        # df = df.sample(weights=df.species_id.apply(
        #     lambda x: self.inv_counts[x]))
        fn = osp.join(self.data_root, f"{song_id}.mp3") # This is the file name
        # df = df.squeeze()
        # t0 = max(df['t_min'], 0)
        # t1 = max(df['t_max'], 0)
        # t0 = np.random.uniform(t0, t1)
        # t0 = max(t0, 0)
        # t0 = min(t0, self.total_duration-self.duration)
        # t1 = t0 + self.duration
        # valid_df = self.tp[self.tp.recording_id == recording_id]
        # valid_df = valid_df[(valid_df.t_min < t1) & (valid_df.t_max > t0)]
        y, _ = librosa.load(fn, sr=None)

        y = self.resampler(torch.from_numpy(y).float()).numpy()
        y = self.transform(y)
        if random.random() < 0.25:
            tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr)
            y = librosa.clicks(frames=beats, sr=self.sr, length=len(y))

        melspec = librosa.feature.melspectrogram(
            y, sr=self.sr, n_mels=self.nmels, fmin=self.fmin, fmax=self.fmax,
        )
        melspec = librosa.power_to_db(melspec)
        melspec = mono_to_color(melspec)
        melspec = normalize(melspec, mean=None, std=None)
        melspec = self.img_transform(image=melspec)['image']
        melspec = np.moveaxis(melspec, 2, 0)
        return melspec, labels