import os
import argparse
import tensorflow.compat.v1 as tf
from SSAH import SSAH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='flickr', help='Dataset name: flickr/coco/nuswide')
    parser.add_argument('--bits', type=int, default=32, help='16/32/64/128')
    parser.add_argument('--epochs', type=int, default=500, help='The epoch of training stage.')
    parser.add_argument('--phase', type=str, default='train', help='"train": training and test/ "test": load chechpoint and test')
    # parser.add_argument('--gpuID', type=int, default='0', help='The epoch of training stage.')
    config = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = config.gpuID

    # gpuconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction))
    gpuconfig = tf.ConfigProto()

    with tf.Session(config=gpuconfig) as sess:
        model = SSAH(sess, config)
        model.train() if config.phase == 'train' else model.test('test')

