from librosa import feature
import spleeter
import pandas as pd
import os
import librosa
import eyed3
import uuid
import tqdm
import logging
import numpy as np
from spleeter.separator import Separator
                
                
eyed3.log.setLevel("ERROR")


class Preprocessing_Pipeline():

    def __init__(self,config) -> None:
        
        self.unsplit_path=config['Splitting']['unsplit_data_root']
        # self.split_path = config['Splitting']['split_data_root']
        self.unsplit_dataset_path = config['Splitting']['unsplit_dataset_path']
        self.separator = Separator('spleeter:2stems')
        self.unsplit_dataset = dataset=pd.DataFrame(columns=['artist','song','album','year','genre','uuid','duration','rms'])
        

    def split_song(self,target_path):
        pass

    def split_folder(self,target_path):
        pass

    def build_unsplit_dataset(self):
        # dataset=pd.DataFrame(columns=['artist','song','album','year','genre','uuid','duration','bpm','chroma_stft','rmse','speftral_centroid','spectral_bandwidth','rolloff','zero_crossing_rate'])
        
        
        for artist in os.listdir(self.unsplit_path):
            print(artist)
            for song in os.listdir(self.unsplit_path+f"{artist}/"):
                try:
                    feature_row=[]
                    a = eyed3.load(self.unsplit_path+f"{artist}/"+song)
                    feature_row.append(artist)
                    feature_row.append(a.tag.title)
                    feature_row.append(a.tag.album)
                    feature_row.append(str(a.tag.getBestDate()))
                    feature_row.append(a.tag.genre.name)
                    feature_row.append(str(uuid.uuid4()))
                    feature_row.append(a.info.time_secs)
                    
                    y, sr = librosa.load(self.unsplit_path+f'{artist}/'+song, mono=True)
                    feature_row.append(np.mean(librosa.feature.rms(y=y)))
                    
                    # feature_row.append(librosa.feature.chroma_stft(y=y, sr=sr))
                    # feature_row.append(librosa.feature.rms(y=y))
                    # feature_row.append(librosa.feature.spectral_centroid(y=y, sr=sr))
                    # feature_row.append(librosa.feature.spectral_bandwidth(y=y, sr=sr))
                    # feature_row.append(librosa.feature.spectral_rolloff(y=y, sr=sr))
                    # feature_row.append(librosa.feature.zero_crossing_rate(y))
                    self.unsplit_dataset=self.unsplit_dataset.append(feature_row)
                    print(feature_row)
                except Exception as error:
                    print(error)
                    pass
        self.unsplit_dataset.to_csv(self.unsplit_dataset_path)


    def run_pipeline(self):
        self.build_unsplit_dataset()
        
