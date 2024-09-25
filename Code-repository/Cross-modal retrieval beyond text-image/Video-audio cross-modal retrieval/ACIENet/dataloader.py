import torch
import torch.nn.functional
from torch.utils.data import Dataset
from data import *


class train_data_V2F(Dataset):
    def __init__(self):
        Fusion_list = get_anchor_audio(0)
        print('train_samples:', len(Fusion_list))
        self.data = Fusion_list

    def __getitem__(self, item):
        a, f1, f2, ID, a2, label, sex1, sex2, sex3, sex4, nat1, nat2, nat3, nat4 = self.data[item]
        label = int(label)
        ID = int(ID)
        a = load_audio(a)
        a = torch.from_numpy(a)

        f1 = TransformToPIL(f1)
        f1 = target_transform_train(f1)

        f2 = TransformToPIL(f2)
        f2 = target_transform_train(f2)

        a2 = load_audio(a2)
        a2 = torch.from_numpy(a2)

        face_m = 0 
        audio_m = 1

        sex1, sex2, sex3, sex4 = int(sex1), int(sex2), int(sex3), int(sex4)
        nat1, nat2, nat3, nat4 = int(nat1), int(nat2), int(nat3), int(nat4)
        Att_a = 3*sex1 + nat1
        Att_f1 = 3*sex2 + nat2
        Att_f2 = 3*sex3 + nat3
        Att_a2 = 3*sex4 + nat4

        return a, f1, f2, a2, ID, label, face_m, audio_m, Att_a, Att_f1, Att_f2, Att_a2

    def __len__(self):
        return len(self.data)


class test_data_V2F(Dataset):
    def __init__(self):
        Fusion_list = get_anchor_audio(1)
        print('test_samples:', len(Fusion_list))
        self.data = Fusion_list

    def __getitem__(self, item):
        a, f1, f2, ID, label = self.data[item]
        label = int(label)

        a = load_audio(a)
        a = torch.from_numpy(a)

        f1 = TransformToPIL(f1)
        f1 = target_transform_test(f1)

        f2 = TransformToPIL(f2)
        f2 = target_transform_train(f2)

        return a, f2, f1, label

    def __len__(self):
        return len(self.data)
