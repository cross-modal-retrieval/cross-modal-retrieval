import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset

from utils.datasets import *

class DatasetProcessing(Dataset):
    def __init__(self, dataname, mode, transform=None):
        self.dataname = dataname
        self.transform = transform
        self.mode = mode

        dset = load_dataset(mode=mode, dataname=self.dataname)
        self.label = dset.label
        self.image = dset.img_feature
        dset = 0

    def __getitem__(self, index):
        img = self.image[index] 
        if self.transform is not None:
            # img = img[:, :, ::-1] # BGR -> RGB
            img = Image.fromarray(img)
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label, index

    def __len__(self):
        return len(self.label)

class DatasetProcessing_txt(Dataset):
    def __init__(self, dataname, mode, transform=None):
        self.dataname = dataname
        self.transform = transform
        self.mode = mode

        dset = load_dataset(mode=mode, dataname=dataname)
        self.label = dset.label
        self.text = dset.txt_feature
        self.y_dim = self.text.shape[1]
        dset = 0

    def __getitem__(self, index):
        vector = self.text[index, :]
        label = torch.from_numpy(self.label[index])
        return vector, label, index

    def __len__(self):
        return len(self.label)
