from numpy.core.numeric import True_
import torch
import numpy as np
import os.path as osp

from torch.utils import data
from src.audio_augmentations import AudioTransform, Compose, OneOf, GaussianNoiseSNR, PinkNoiseSNR, TimeShift, VolumeControl
from src.spectrogram_utils import mono_to_color, normalize
import torchaudio
import sklearn
import os

from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset
import pandas as pd
import librosa
import albumentations as A
import random
import librosa.display
import matplotlib.pyplot as plt

import torch
from torch.utils.data import dataloader, Subset, DataLoader
import torch.nn as nn
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from resnest.torch.resnet import ResNet, Bottleneck
from torchvision.models import resnet18, resnet34, resnet50
from resnest.torch import resnest50

from sklearn.model_selection import train_test_split
# from torch.functional import cr

import warnings
warnings.filterwarnings("ignore")


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


class RecoNetDataset(Dataset):
    def __init__(self, tp, config=None):
        self.tp = tp
        self.config = config
        self.sr = self.config.sr
        self.data_root = self.config.TRAIN_AUDIO_ROOT
        self.nmels = self.config.nmels
        self.fmin, self.fmax = 40, 20000
        self.num_classes = self.tp.artist.nunique()
        self.classes = list(self.tp.artist.unique())
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

    def __len__(self):
        return len(self.tp)

    def __getitem__(self, idx):
        # Number of artists here
        labels = np.zeros((self.num_classes,), dtype=np.float32)

        song_id = self.tp.loc[idx, 'new_uuid']  # Get the song ID here
        df = self.tp.loc[self.tp.new_uuid == song_id]
        artist = df.artist.unique()[0]
        # print(artist)
        labels[self.classes.index(artist)] = 1
        # print(labels)
        # This is the file name
        fn = osp.join(self.data_root, f"{song_id}.mp3")
        # print(fn)
        y, _ = librosa.load(fn, sr=self.sr)

        y = self.transform(y)
        if random.random() < 0.25:
            tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr)
            y += librosa.clicks(frames=beats, sr=self.sr, length=len(y))

        melspec = librosa.feature.melspectrogram(
            y, sr=self.sr, n_mels=self.nmels, fmin=self.fmin, fmax=self.fmax,
        )
        # fig, ax = plt.subplots()
        melspec = librosa.power_to_db(melspec)
        # img = librosa.display.specshow(melspec, x_axis='time',
        #    y_axis='mel', sr=self.sr, ax=ax, cmap='viridis')

        melspec = mono_to_color(melspec)
        melspec = normalize(melspec, mean=None, std=None)
        melspec = self.img_transform(image=melspec)['image']
        melspec = np.moveaxis(melspec, 2, 0)
        return melspec, labels

class RecoNetTestDataset(Dataset):
    def __init__(self, tp, config=None):
        self.tp = tp
        self.config = config
        self.sr = self.config.sr
        self.data_root = self.config.TRAIN_AUDIO_ROOT
        self.nmels = self.config.nmels
        self.fmin, self.fmax = 40, 20000
        self.num_classes = self.tp.artist.nunique()
        self.classes = list(self.tp.artist.unique())
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_mels=self.nmels,
                                                        f_min=self.fmin, f_max=self.fmax,
                                                        n_fft=2048)

    def __len__(self):
        return len(self.tp)

    def __getitem__(self, idx):
        # Number of artists here
        labels = np.zeros((self.num_classes,), dtype=np.float32)

        song_id = self.tp.loc[idx, 'new_uuid']  # Get the song ID here
        df = self.tp.loc[self.tp.new_uuid == song_id]
        artist = df.artist.unique()[0]
        # print(artist)
        labels[self.classes.index(artist)] = 1
        # print(labels)
        # This is the file name
        fn = osp.join(self.data_root, f"{song_id}.mp3")
        # print(fn)
        y, _ = librosa.load(fn, sr=self.sr)

        melspec = librosa.feature.melspectrogram(
            y, sr=self.sr, n_mels=self.nmels, fmin=self.fmin, fmax=self.fmax,
        )
        # fig, ax = plt.subplots()
        melspec = librosa.power_to_db(melspec)
        # img = librosa.display.specshow(melspec, x_axis='time',
        #    y_axis='mel', sr=self.sr, ax=ax, cmap='viridis')

        melspec = mono_to_color(melspec)
        melspec = normalize(melspec, mean=None, std=None)
        melspec = np.moveaxis(melspec, 2, 0)
        return melspec, labels

class RecoNetTrainer:

    def __init__(self, batch_size):
        self.model_configs = {
            "resnest50_fast_1s1x64d":
            {
                "num_classes": 264,
                "block": Bottleneck,
                "layers": [3, 4, 6, 3],
                "radix": 1,
                "groups": 1,
                "bottleneck_width": 64,
                "deep_stem": True,
                "stem_width": 32,
                "avg_down": True,
                "avd": True,
                "avd_first": True
            }
        }
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.get_model().to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.configure_optimizers()
        self.train_losses = []
        self.val_losses = []
        self.accuracy = accuracy_score
        self.train_accuracy = []
        self.val_accuracy = []
        self.batch_size = batch_size
        

    def get_model(self, pretrained=True, n_class=23):
        # model = torchvision.models.resnext50_32x4d(pretrained=False)
        # model = torchvision.models.resnext101_32x8d(pretrained=False)
        model = ResNet(**self.model_configs["resnest50_fast_1s1x64d"])
        fn = 'pretrained_models/resnest50_fast_1s1x64d_conf_1.pt'
        model.load_state_dict(torch.load(fn))
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, 23)
        model.fc = nn.Linear(n_features, n_class)
        return model

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(), lr=0.001,
                                  weight_decay=0.0001)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                                    mode='min',
                                                                    factor=0.5,
                                                                    patience=2,
                                                                    verbose=True),
            'monitor': 'validation_loss',
            'interval': 'epoch',
            'frequency': 1,
            'strict': True,
        }

        self.optimizer = optim
        self.scheduler = scheduler

        return [optim], [scheduler]

    def generate_dataloaders(self, train_dataset,val_dataset):
        test_indices, val_indices, _, _ = train_test_split(
            range(len(val_dataset)),
            range(len(val_dataset)),
            test_size=0.5)

        test_split   = Subset(val_dataset, test_indices)
        val_split = Subset(val_dataset, val_indices)
        train_split = train_dataset

        self.train_batches = DataLoader(
            train_split, batch_size=self.batch_size, shuffle=True)
        self.val_batches = DataLoader(
            val_split, batch_size=self.batch_size, shuffle=True)
        self.test_batches = DataLoader(
            test_split, batch_size=self.batch_size, shuffle=True)
        self.test_loader_one = DataLoader(test_split, batch_size=1, shuffle=True_, drop_last=True)

    def train_step(self, x, y):
        self.model.train()
        preds = self.model(x)
        loss = self.loss_fn(preds, torch.argmax(y, dim=-1).to(self.device))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def val_step(self, x, y):
        self.model.eval()
        preds = self.model(x)
        loss = self.loss_fn(preds, torch.argmax(y, dim=-1).to(self.device))
        return loss.item()



    def train(self, n_epochs):
        t = trange(1, n_epochs + 1)
        for epoch in t:
            batch_losses = []
            batch_accuracy = []
            for x_batch, y_batch in self.train_batches:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(float(loss))
                batch_accuracy.append(self.accuracy(torch.argmax(
                    self.model(x_batch), dim=-1).cpu(), torch.argmax(y_batch, dim=-1).cpu()))
            training_loss = np.mean(batch_losses)
            training_accuracy = np.mean(batch_accuracy)
            self.train_losses.append(training_loss)
            self.train_accuracy.append(training_accuracy)
            del loss

            with torch.no_grad():
                batch_val_losses = []
                batch_val_acc = []
                for x_val, y_val in self.val_batches:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    self.model.eval()
                    val_loss = self.val_step(x_val, y_val)
                    batch_val_losses.append(float(val_loss))
                    batch_val_acc.append(self.accuracy(torch.argmax(
                        self.model(x_val), dim=-1).cpu(), torch.argmax(y_val, dim=-1).cpu()))
                validation_loss = np.mean(batch_val_losses)
                validation_accuracy = np.mean(batch_val_acc)
                self.val_losses.append(validation_loss)
                self.val_accuracy.append(validation_accuracy)
            del val_loss

            t.set_description(
                f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f} \t train_acc: {training_accuracy:.4f}\t val_acc: {validation_accuracy:.4f}")
            t.refresh()
            torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'loss': training_loss,
            'training_losses' : self.train_losses,
            'val_losses'  : self.val_losses,
            'train_acc' : self.train_accuracy,
            'val_acc' : self.val_accuracy,
            'train_loader' : self.train_batches,
            'val_loader' : self.val_batches,
            'test_loader' : self.test_batches,
            'test_loader_one' : self.test_loader_one
            }, '{}_{}.pt'.format(int(datetime.datetime.now().timestamp()),int(validation_accuracy*100)))

    def evaluate(self, batch_size=1):
        
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in self.test_loader_one:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values

    def load_model_checkpoint(self,path,all=True):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if all:
            self.train_losses = checkpoint['training_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accuracy = checkpoint['train_acc']
            self.val_accuracy = checkpoint['val_acc']
            self.train_batches = checkpoint['train_loader']
            self.val_batches = checkpoint['val_loader']
            self.test_batches = checkpoint['test_loader']
            self.test_loader_one = checkpoint['test_loader_one']


    def plot_losses(self, ax=None):
        if ax is None:
            plt.plot(self.train_losses, label="Training loss")
            plt.plot(self.val_losses, label="Validation loss")
            plt.legend()
            plt.title("Losses")
            plt.show()
            plt.close()
        else:
            ax.plot(self.train_losses, label="Training loss")
            ax.plot(self.val_losses, label="Validation loss")
            ax.legend()

    def score(self, test_loader):
        preds = self.model(test_loader.data)
        return {'acc': accuracy_score(preds, test_loader.targets),
                'recall': recall_score(preds, test_loader.targets),
                'prec': precision_score(preds, test_loader.targets),
                'F1': f1_score(preds, test_loader.targets)}
