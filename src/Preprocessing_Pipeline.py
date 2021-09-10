from unicodedata import name
from librosa import feature
import spleeter
import pandas as pd
import os
import librosa
import eyed3
import uuid
from spleeter.types import AudioDescriptor
from tqdm import tqdm
import logging
import numpy as np
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import tempfile
import subprocess
import shutil
import time
import random as rd
from faker import Faker
from pydub import AudioSegment
from pydub.silence import split_on_silence
import pydub
from tqdm import trange


eyed3.log.setLevel("ERROR")


def simple_genre(x):
    if x in ['Europop/Synth-Pop/Rhythm & Blues', 'Pop', 'Pop, R&B', 'ポップ', 'Asian Music']:
        return 'Pop'
    if x in ['Rnb/Swing', 'Contemporary R&B', "R&B/Pop", "R&B, Reggae", "R&B/Soul"]:
        return 'R&B'
    if x in ['Trap/Rnb/Swing', 'Rap/Hip Hop', 'Hip Hop/Rap', 'Hip-Hop/Rap', 'Rap', "Rap & Hip-Hop", "Hip-Hop", "Rap/Hip Hop;Pop"]:
        return "Hip Hop"
    if x in ['Afrikaanse Muziek', 'Dance']:
        return 'Unknown'
    if x == 'Soul And R&B' or x == 'Funk':
        return 'Soul'
    if x == 'Pop-RnB':
        return "R&B"
    else:
        return x


class Splitting_Pipeline():

    def __init__(self, config) -> None:

        self.unsplit_path = config['Splitting']['unsplit_data_root']
        self.split_path = config['Splitting']['split_data_root']
        self.unsplit_dataset_path = config['Splitting']['unsplit_dataset_path']
        self.whole_spectrogram_root = config['Spectrogram']['full_spectrogram_data_root']
        self.split_pool = config['Splitting']['split_pool_path']
        self.separator = Separator('spleeter:2stems')
        columns = ['artist', 'song', 'album', 'year',
                   'genre', 'uuid', 'duration', 'rms', 'path']
        self.unsplit_dataset = pd.DataFrame(columns=columns)

    def split_song(self, current_path, target_path, song):
        try:
            song_id = self.unsplit_dataset[self.unsplit_dataset.path ==
                                           song].uuid.iloc[0]
            if not os.path.exists(target_path+'/'+song[0:len(song)-4]+'/'+'vocals.mp3'):
                self.separator.separate_to_file(
                    current_path, destination=target_path, synchronous=False, bitrate='320k', codec='mp3')
            print(song)
            while not os.path.exists(target_path+'/'+song[0:len(song)-4]+'/'+'vocals.mp3'):
                time.sleep(1)

            if not os.path.exists(self.split_pool+f'{song_id}.mp3'):
                shutil.copyfile(
                    target_path+'/'+song[0:len(song)-4]+'/'+'vocals.mp3', self.split_pool+f'{song_id}.mp3')
        except:
            pass

    def split_folder(self, artist_path, target_path):
        for song in os.listdir(artist_path):
            self.split_song(f'{artist_path}/'+song, target_path, song)

    def split_dataset(self):
        for artist in os.listdir(self.unsplit_path):
            if not os.path.exists(self.split_path+f'{artist}'):
                os.makedirs(self.split_path+f'{artist}')
            self.split_folder(self.unsplit_path +
                              f'{artist}', self.split_path+f'{artist}')

    def build_unsplit_dataset(self):
        # dataset=pd.DataFrame(columns=['artist','song','album','year','genre','uuid','duration','bpm','chroma_stft','rmse','speftral_centroid','spectral_bandwidth','rolloff','zero_crossing_rate'])
        f1 = Faker()
        count = 0
        for artist in os.listdir(self.unsplit_path):
            print(artist)

            for k in tqdm(range(len(os.listdir(self.unsplit_path+f"{artist}/")))):

                count += 1
                Faker.seed(count)
                song = os.listdir(self.unsplit_path+f"{artist}/")[k]
                feature_row = []
                a = eyed3.load(self.unsplit_path+f"{artist}/"+song)
                feature_row.append(artist)
                try:
                    feature_row.append(a.tag.title)
                except:
                    feature_row.append(song)
                try:
                    feature_row.append(a.tag.album)
                except:
                    feature_row.append('Unknown')
                try:
                    feature_row.append(str(a.tag.getBestDate())[:4])
                except:
                    feature_row.append('None')
                try:
                    if a.tag.genre is not None:
                        feature_row.append(a.tag.genre.name)
                    else:
                        feature_row.append('Unknown')
                except:
                    feature_row.append('Unknown')
                feature_row.append(str(f1.uuid4()))
                try:
                    feature_row.append(a.info.time_secs)
                except:
                    feature_row.append(0)

                feature_row.append(0)
                feature_row.append(song)

                self.unsplit_dataset = self.unsplit_dataset.append(
                    pd.DataFrame([feature_row], columns=self.unsplit_dataset.columns))
        if len(self.unsplit_dataset) > 0:
            self.unsplit_dataset.year = self.unsplit_dataset.year.apply(
                lambda x: x[0:4])
            self.unsplit_dataset.genre = self.unsplit_dataset.genre.apply(
                simple_genre)
            self.unwanted_uuids = self.unsplit_dataset[self.unsplit_dataset.duration <= 30].uuid
            print(f'filtered {len(self.unwanted_uuids)} songs')
            self.unsplit_dataset = self.unsplit_dataset[self.unsplit_dataset.duration > 30]
            self.unsplit_dataset.to_csv(self.unsplit_dataset_path, index=False)

    def clean_unused_songs(self):
        temp = 0
        for song in os.listdir(self.split_pool):
            if song.split('.')[0] in self.unwanted_uuids:
                os.remove(self.split_pool+song)
                temp += 1
        print(f'removed {temp} songs')

    def run_pipeline(self, split, build_dataset):
        if build_dataset:
            self.build_unsplit_dataset()
        if split:
            self.split_dataset()
            self.clean_unused_songs()


class SegmentPipeline:
    def __init__(self, config) -> None:

        self.split_pool = config['Splitting']['split_pool_path']
        self.database = pd.read_csv('data/data/unsplit_dataset.csv')
        self.segmented_pool = config['Segmenting']['segmented_data_root']
        self.segment_length = config['Segmenting']['segment_length']*1000

    def split_song_silence(self, filepath):
        song = AudioSegment.from_mp3(filepath)
        chunks = [song[5000:len(song)-5000]]
        # chunks = split_on_silence(
        #     # Use the loaded audio.
        #     song,
        #     min_silence_len=5000,
        #     silence_thresh=-20,
        #     keep_silence=100
        # )
        # if len(chunks)==0:
        #     chunks = [song]
        return chunks
    
    def split_song_chunks(self,filepath):
        total_chunks=[]
        chunks = self.split_song_silence(filepath)
        for chunk in chunks:
            total_chunks +=  pydub.utils.make_chunks(chunk, self.segment_length)
        for k in range( len(total_chunks)):
            chunk = self.pad_audio(total_chunks[k])
            if not os.path.exists('data/data/segmented_pool/{}_{}.mp3'.format(filepath.split('/')[-1].split('.')[0],k)):
                chunk.export('data/data/segmented_pool/{}_{}.mp3'.format(filepath.split('/')[-1].split('.')[0],k), format='mp3')
            
    
    def pad_audio(self,audio):
        if len(audio)==self.segment_length:
            return audio
        else:
            audio = audio + AudioSegment.silent(duration=self.segment_length-len(audio))
            return audio

    def segment_split_pool(self):
        t = trange(len(os.listdir(self.split_pool)))
        for k in t:
            id = os.listdir(self.split_pool)[k].split('/')[-1].split('.')[0]
            song_name = self.database[self.database.uuid==id].iloc[0].artist + ' - ' + str(self.database[self.database.uuid==id].iloc[0].song)
            t.set_description(song_name)
            t.refresh()
            self.split_song_chunks(self.split_pool+os.listdir(self.split_pool)[k])


    