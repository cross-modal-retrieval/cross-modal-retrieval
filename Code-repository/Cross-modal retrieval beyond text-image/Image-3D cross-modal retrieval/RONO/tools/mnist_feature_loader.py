from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np

class FeatureDataloader(Dataset):
    def __init__(self, num_classes=10, partition='train'):
        self.num_classes = num_classes
        if partition == 'train':
            self.img_feat = np.load('datasets/3D_MNIST/train_img_feat.npy')
            self.pt_feat = np.load('datasets/3D_MNIST/train_pt_feat.npy')  
            self.label = np.load('datasets/3D_MNIST/train_label_60.npy')
            self.ori_label = np.load('datasets/3D_MNIST/train_ori_label.npy')
    
    def __getitem__(self, item):
        img_feat =  self.img_feat[item]
        pt_feat =  self.pt_feat[item]
        label = self.label[item]
        ori_label = self.ori_label[item]

        return img_feat, pt_feat, label, ori_label
    def __len__(self):
        return self.label.shape[0]