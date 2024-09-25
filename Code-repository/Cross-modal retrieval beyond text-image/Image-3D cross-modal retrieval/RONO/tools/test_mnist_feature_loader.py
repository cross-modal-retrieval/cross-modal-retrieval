from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch
import numpy as np

class FeatureDataloader(Dataset):
    def __init__(self, num_classes=10, partition='test'):
        self.num_classes = num_classes
        if partition == 'test':
            self.img_feat = np.load('./datasets/3D_MNIST/test_img_feat.npy')
            self.pt_feat = np.load('./datasets/3D_MNIST/3D_MNIST/test_pt_feat.npy')        
            self.ori_label = np.load('./datasets/3D_MNIST/test_ori_label.npy')
    def __getitem__(self, item):
        img_feat =  self.img_feat[item]
        pt_feat =  self.pt_feat[item]
        ori_label = self.ori_label[item]


        return img_feat, pt_feat, ori_label
    def __len__(self):
        return self.ori_label.shape[0]