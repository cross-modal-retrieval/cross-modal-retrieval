from collections import namedtuple
import h5py
import torch
import torch.utils.data as data

paths = {
    'flickr': '../Data/raw_mir.mat',
    'nuswide': '../Data/raw_nus.mat',
    'coco': '../Data/raw_coco.mat'
}

dataset_lite = namedtuple('dataset_lite', ['img_feature', 'txt_feature', 'label'])

def load_dataset(mode, dataname):
    if mode == 'train':
        data = h5py.File(paths[dataname], 'r')
        img_feature = data['I_tr'][:].transpose(3, 0, 1, 2)
        txt_feature = data['T_tr'][:].T
        label = data['L_tr'][:].T
    elif mode == 'retrieval':
        data = h5py.File(paths[dataname], 'r')
        img_feature = data['I_db'][:].transpose(3, 0, 1, 2)
        txt_feature = data['T_db'][:].T
        label = data['L_db'][:].T
    else:
        data = h5py.File(paths[dataname], 'r')
        img_feature = data['I_te'][:].transpose(3, 0, 1, 2)
        txt_feature = data['T_te'][:].T
        label = data['L_te'][:].T

    return dataset_lite(img_feature, txt_feature, label)


class my_dataset(data.Dataset):
    def __init__(self, img_feature, txt_feature, label):
        self.img_feature = torch.Tensor(img_feature)
        self.txt_feature = torch.Tensor(txt_feature)
        self.label = torch.Tensor(label)
        self.length = self.img_feature.size(0)

    def __getitem__(self, item):
        return item, self.img_feature[item, :], self.txt_feature[item, :], self.label[item, :]

    def __len__(self):
        return self.length

