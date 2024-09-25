""" AudioCaps dataset Module.
"""
import copy
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import hickle

import torch 
from torch.utils.data import Dataset, DataLoader


class AudioCaps(Dataset):
    """
    Available audio features:
        - VGGish pretrained feature: vggish
        - Resnet18 VGGSound pretrained feature: vggsound
        - PANNs pretrained feature: panns_cnn10, panns_cnn14
    w2v embedding pretrained by googleNews-vectors-negative300

    :Params: w2v_file: filepath to w2v pickle file 
             audio_h5: filepath to audio_experts, List[str]
             audio_experts: list of audio_experts, List[str]
             filename: filepath to index file
             split: datasplit train, val or test

    """

    def __init__(self, 
                w2v_file,
                audio_h5,
                audio_experts,
                filename,
                max_words,
                audio_padding_length,
                split
                ):
        self.modalities = audio_experts
        self.audio_h5 = audio_h5
        self.num_audio_features = len(audio_h5)
        self.w2v_file = w2v_file
        self._Audiofeaturedataset =[]
        self._w2v = pickle.load(open(w2v_file,'rb'))
        df = pd.read_csv(filename)
        self.fname = df['youtube_id'].unique()
        self.split = split
        self.max_words = max_words
        self.audio_padding_length = audio_padding_length

    def __len__(self):
        return len(self.fname)
    
    def __getitem__(self, idx):
        filename = self.fname[idx]

        # get audio features
        if self._Audiofeaturedataset==[]:
            for i in range(self.num_audio_features):
                self._Audiofeaturedataset.append(h5py.File(self.audio_h5[i],'r')) 

        audio_features = {}
        audio_masks = {}
        for i, mod in enumerate(self.modalities):
            audio_feature = self._Audiofeaturedataset[i][filename][()]
            audio_features[mod] = np.zeros((self.audio_padding_length[mod], audio_feature.shape[1]))
            if audio_feature.shape[0] <= self.audio_padding_length[mod]:
                audio_features[mod][:audio_feature.shape[0],:] = audio_feature
            else:
                audio_features[mod] = audio_feature[:self.audio_padding_length[mod],:]
            audio_masks[mod] = [1] * audio_feature.shape[0]
            while len(audio_masks[mod]) < self.audio_padding_length[mod]:
                audio_masks[mod].append(0)
            if len(audio_masks[mod]) > self.audio_padding_length[mod]:
                audio_masks[mod] = audio_masks[mod][:self.audio_padding_length[mod]]
            assert len(audio_masks[mod]) == self.audio_padding_length[mod]
            assert audio_features[mod].shape[0] == self.audio_padding_length[mod]
            
            audio_features[mod] = torch.from_numpy(audio_features[mod])
            audio_features[mod] = audio_features[mod].float()
            audio_masks[mod] = np.array(audio_masks[mod])
            audio_masks[mod] = torch.from_numpy(audio_masks[mod])

        #get text features
        captions = self._w2v[filename]
        text_features = []
        max_token_masks = []
        for i in range(len(captions)):
            caption = captions[i]
            max_token_masks.append([1] * caption.shape[0])
            text_features.append(np.zeros((self.max_words,caption.shape[-1])))
            if caption.shape[0] <= self.max_words:
                text_features[i][:caption.shape[0]] = caption
            else:
                text_features[i] = caption[:self.max_words,:]
            
            while len(max_token_masks[i]) < self.max_words:
                max_token_masks[i].append(0)
            if len(max_token_masks[i]) > self.max_words:
                max_token_masks[i] = max_token_masks[i][:self.max_words]
            assert len(max_token_masks[i]) == self.max_words
            assert text_features[i].shape[0] == self.max_words
            
            text_features[i] = torch.from_numpy(text_features[i])
            text_features[i] = text_features[i].float()
            max_token_masks[i] = np.array(max_token_masks[i])
            max_token_masks[i] = torch.from_numpy(max_token_masks[i])
        
        text_features = torch.cat(text_features)
        max_token_masks = torch.cat(max_token_masks)
        text_features = text_features.view(len(captions),self.max_words, -1)
        max_token_masks = max_token_masks.view(len(captions), self.max_words)
        ind = {mod: torch.ones(1) for mod in self.modalities}
        return {'experts': audio_features,
                'text': text_features,
                'expert_masks': audio_masks,
                'text_token_masks': max_token_masks,
                'ind':ind,
                }

def create_train_dataloader(
                w2v_file,
                audio_h5,
                audio_experts,
                filename,
                max_words,
                audio_padding_length,
                split,
                batch_size = 128):
    train_dataset = AudioCaps(w2v_file = w2v_file,
                              audio_h5 = audio_h5,
                              audio_experts = audio_experts,
                              filename = filename,
                              max_words=max_words,
                              audio_padding_length=audio_padding_length,
                              split = "train")
    trainloader = DataLoader(train_dataset, 
                             batch_size=batch_size,
                             shuffle=True)
    return trainloader

def create_val_dataloader(
                w2v_file,
                audio_h5,
                audio_experts,
                filename,
                max_words,
                audio_padding_length,
                split = "val"):
    val_dataset = AudioCaps(w2v_file = w2v_file,
                              audio_h5 = audio_h5,
                              audio_experts = audio_experts,
                              filename = filename,
                              max_words=max_words,
                              audio_padding_length=audio_padding_length,
                              split = split)
    batch_size = len(val_dataset)
    valloader = DataLoader(val_dataset, 
                             batch_size=batch_size,
                             shuffle=False)
    return valloader


