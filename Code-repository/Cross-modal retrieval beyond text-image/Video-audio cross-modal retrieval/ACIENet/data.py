import torchvision.transforms as transforms
import librosa
import numpy as np
from PIL import Image


def target_transform_train(PILImg):
    transf = transforms.Compose([
        transforms.Pad(20),
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transf(PILImg)

def target_transform_test(PILImg):
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transf(PILImg)


def TransformToPIL(PILimage):
    RealPILimage = Image.open(PILimage).convert("RGB")
    RealPILimage = RealPILimage.resize((224, 224))
    return RealPILimage


def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def load_audio(audiopath):
    y, sr = librosa.load(audiopath)
    y = y - y.mean()
    y = preemphasis(y)
    y = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    a = np.max(y)
    b = np.min(y)
    k = 2 / (a - b)
    y = -1 + (y - b) * k
    data = y.reshape(y.shape[0]*y.shape[1],1)
    if data.shape[0] < 16000*10:
            y = np.tile(data, (16000*10 + data.shape[0] - 1) // data.shape[0])
    y = np.reshape(data[:16000*10], [-1]).astype(np.float32)
    
    return y



def get_anchor_audio(flag):
    FusionList_anchor_audio = []
    if flag == 0:
        with open('train_V2F_8vox1_label50.txt', 'r') as f:
            for line in f:
                str = line.split()
                FusionList_anchor_audio.append((str[0],) + (str[1],) + (str[2],) + (str[3],) + (str[4],) + (str[5],) + (str[6],) + (str[7],) + (str[8],) + (str[9],) + (str[10],) + (str[11],) + (str[12],) + (str[13],))  
    elif flag == 1:
        #with open('test_V2F_8vox1_label50.txt', 'r') as f:
        with open('test_V2_V2F_8vox1_label50.txt', 'r') as f:
            for line in f:
                str = line.split()
                FusionList_anchor_audio.append((str[0],) + (str[1],) + (str[2],) + (str[3],) + (str[4],))

    return FusionList_anchor_audio
