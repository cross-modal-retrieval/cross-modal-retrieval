import h5py
from PIL import Image
import torch
from torchvision import transforms

from args import config

# RGB
all_data = h5py.File(config.DIR, 'r')

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),  # HWC+[0,255] -> CHW+[0,1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

txt_feat_len = all_data['T_tr'].shape[0]

class MY_DATASET(torch.utils.data.Dataset):
    def __init__(self, transform=None, target_transform=None, train=True, database=False):
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.train_labels = all_data['L_tr'][:].T
            self.txt = all_data['T_tr'][:].T
            self.images = all_data['I_tr'][:].transpose(3, 0, 1, 2)
        elif database:
            self.train_labels = all_data['L_db'][:].T
            self.txt = all_data['T_db'][:].T
            self.images = all_data['I_db'][:].transpose(3, 0, 1, 2)
        else:
            self.train_labels = all_data['L_te'][:].T
            self.txt = all_data['T_te'][:].T
            self.images = all_data['I_te'][:].transpose(3, 0, 1, 2)

    def __getitem__(self, index):

        img, target = self.images[index, :, :, :], self.train_labels[index]
        # img = img[:, :, ::-1].copy()  # BGR -> RGB
        # img = Image.fromarray(np.transpose(img, (2, 1, 0))) # HWC
        img = Image.fromarray(img)
        txt = self.txt[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, txt, target, index

    def __len__(self):
        return len(self.train_labels)
