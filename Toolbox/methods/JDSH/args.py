import argparse
import json

from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description='JDSH')
parser.add_argument('--Train', default=True, help='train or test', type=bool)
parser.add_argument('--dataname', default='COCO', help='flickr/nuswide/coco', type=str)
parser.add_argument('--Checkpoint', default='MIRFlickr_BIT_128.pth', help='checkpoint name', type=str)
parser.add_argument('--bits', default=128, help='hash bit', type=int)
args = parser.parse_args()

Config = './config/JDSH.json'

# load basic settings
with open(Config, 'r') as f:
    config = edict(json.load(f))

# update settings
config.TRAIN = args.Train
config.CHECKPOINT = args.Checkpoint
config.HASH_BIT = args.bits
config.DATASET_NAME = args.dataname
config.FLAG_savecode = False

if args.dataname == 'flickr':
    config.DIR = '../Data/raw_mir.mat'
elif args.dataname == 'nuswide':
    config.DIR = '../Data/raw_nus.mat'
elif args.dataname == 'coco':
    config.DIR = '../Data/raw_coco.mat'  
else:
    print('Dataname Error!')
    config.DIR = ''