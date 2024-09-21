import numpy as np
import scipy.io
import sys

# environmental setting: setting the following parameters based on your experimental environment.
per_process_gpu_memory_fraction = 0.9

# Initialize data loader
MODEL_DIR = '../models/imagenet-vgg-f.mat'

phase = 'train' # TODO
checkpoint_dir = './checkpoint'
savecode_dir = './Hashcode'
result_dir = './result'

netStr = 'alex'

SEMANTIC_EMBED = 512
MAX_ITER = 100
batch_size = 128
image_size = 224

data = scipy.io.loadmat(MODEL_DIR)
mean = data['normalization'][0][0][0]

k_lab_net = 2 # 10
k_img_net = 4 # 15
k_txt_net = 2 # 15
k_dis_net = 1 # 1
save_freq = 1000

alpha = 1
gamma = 1
beta = 1
eta = 1
delta = 1

# Learning rate
lr_lab = [np.power(0.1, x) for x in np.arange(2.0, MAX_ITER, 0.5)]
lr_img = [np.power(0.1, x) for x in np.arange(4.5, MAX_ITER, 0.5)]
lr_txt = [np.power(0.1, x) for x in np.arange(3.5, MAX_ITER, 0.5)]
lr_dis = [np.power(0.1, x) for x in np.arange(3.0, MAX_ITER, 0.5)]
