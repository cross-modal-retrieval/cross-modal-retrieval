import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import yaml
import argparse
import utils
from PIL import Image
import librosa
import scipy.io as scio

def audio_extract(wav_file,sr=16000):
    wav=librosa.load(wav_file,sr=sr)[0]
    # Takes a waveform(length 160000,sampling rate 16000) and extracts filterbank features(size 400*64)
    spec=librosa.core.stft(wav,n_fft=4096,hop_length=200,win_length=1024,window="hann",center=True,pad_mode="constant")
    mel=librosa.feature.mfcc(S=np.abs(spec), sr=sr,n_mfcc=64)#melspectrogram(S=np.abs(spec),sr=sr,n_mels=64,fmax=8000)
    #print(mel.shape)
    logmel=librosa.core.power_to_db(mel[:,:300])#300
    if logmel.shape[1]!=300:
       logmel=np.column_stack([logmel,[[0]*(300-int(logmel.shape[1]))]*64])
    return logmel.T.astype("float32")

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, opt):
        self.loc = opt['dataset']['data_path']
        self.img_path = opt['dataset']['image_path']
        self.aud_path = opt['dataset']['audio_path']
        self.aud_mat_path = opt['dataset']['audio_mat_path']
        self.audios = []
       
        if data_split != 'test':
            aud_mat=scio.loadmat(self.aud_mat_path+"train_audios.mat")
            with open(self.loc+'%s_auds_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    #self.audios.append(line.strip())
                    #aud=audio_extract(self.aud_path  +str(line.strip())[2:-1])
                    aud=aud_mat[str(line.strip())[2:-1]]
                    aud=torch.FloatTensor(aud).unsqueeze(0)
                    self.audios.append(aud)
                    
            self.images = []
            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        else:
            aud_mat=scio.loadmat(self.aud_mat_path+"test_audios.mat")
            with open(self.loc + '%s_auds.txt' % data_split, 'rb') as f:
                for line in f:
                    #self.audios.append(line.strip())
                    #aud=audio_extract(self.aud_path  +str(line.strip())[2:-1])
                    aud=aud_mat[str(line.strip())[2:-1]]
                    
                    aud=torch.FloatTensor(aud).unsqueeze(0)
                    self.audios.append(aud)

            self.images = []
            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        
        self.length = len(self.audios)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((278, 278)),
                transforms.RandomRotation((0, 90)),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        audio = self.audios[index]
        

        
        #audio = audio_extract(self.aud_path  +str(self.audios[index])[2:-1])
        #audio=torch.FloatTensor(audio).unsqueeze(0)
        
        

        image = Image.open(self.img_path  +str(self.images[img_id])[2:-1]).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])
        
        return image, audio, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):

    # Sort a data list by caption length
    
    images, audios, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    audios = torch.stack(audios, 0)
    
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    
    return images, audios, ids


def get_precomp_loader(data_split, batch_size=100,
                       shuffle=True, num_workers=0, drop_last=False, opt={}):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_split, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers,
                                              drop_last=drop_last)
    return data_loader

def get_loaders(opt):
    train_loader = get_precomp_loader( 'train',
                                      opt['dataset']['batch_size'], True, opt['dataset']['workers'], True, opt=opt)
    val_loader = get_precomp_loader( 'val',
                                    opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], True, opt=opt)
    return train_loader, val_loader


def get_test_loader(opt):
    test_loader = get_precomp_loader( 'test',
                                      opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], False, opt=opt)
    return test_loader
