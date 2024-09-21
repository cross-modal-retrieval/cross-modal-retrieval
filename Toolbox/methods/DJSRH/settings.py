import logging
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='flickr', help='Dataset name: flickr/coco/nuswide')
parser.add_argument('--bits', type=int, default=32, help='16/32/64/128')
parser.add_argument('--epochs', type=int, default=500, help='The epoch of training stage.')
# parser.add_argument('--gpuID', type=str, default='0', help='The GPU ID')
args = parser.parse_args()

if args.dataname == 'flickr':
    DIR = '../Data/raw_mir.mat'
elif args.dataname == 'nuswide':
    DIR = '../Data/raw_nus.mat'
elif args.dataname == 'coco':
    DIR = '../Data/raw_coco.mat'  
else:
    print('Dataname Error!')
    DIR = ''

DATASET_NAME = args.dataname
CODE_LEN = args.bits 
# GPU_ID = args.gpuID
FLAG_savecode = False

BETA = 0.6
LAMBDA1 = 0.1
LAMBDA2 = 0.1
NUM_EPOCH = 80
#NUM_EPOCH = 1

LR_IMG = 0.001
LR_TXT = 0.01
EVAL_INTERVAL = 40

BATCH_SIZE = 128 # 32
MU = 1.5
ETA = 0.4

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

NUM_WORKERS = 1
EPOCH_INTERVAL = 2

MODEL_DIR = './checkpoint'
EVAL = False # EVAL = True: just test, EVAL = False: train and eval

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S",time.localtime(time.time())) 
log_name = now + '_log.txt'
log_dir = './log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
txt_log = logging.FileHandler(os.path.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)


logger.info('--------------------------Current Settings--------------------------')
logger.info('EVAL = %s' % EVAL)
logger.info('DATASET_NAME = %s' % DATASET_NAME)
logger.info('BETA = %.4f' % BETA)
logger.info('LAMBDA1 = %.4f' % LAMBDA1)
logger.info('LAMBDA2 = %.4f' % LAMBDA2)
logger.info('NUM_EPOCH = %d' % NUM_EPOCH)
logger.info('LR_IMG = %.4f' % LR_IMG)
logger.info('LR_TXT = %.4f' % LR_TXT)
logger.info('BATCH_SIZE = %d' % BATCH_SIZE)
logger.info('CODE_LEN = %d' % CODE_LEN)
logger.info('MU = %.4f' % MU)
logger.info('ETA = %.4f' % ETA)
logger.info('MOMENTUM = %.4f' % MOMENTUM)
logger.info('WEIGHT_DECAY = %.4f' % WEIGHT_DECAY)
# logger.info('GPU_ID =  %d' % GPU_ID)
logger.info('NUM_WORKERS = %d' % NUM_WORKERS)
logger.info('EPOCH_INTERVAL = %d' % EPOCH_INTERVAL)
logger.info('EVAL_INTERVAL = %d' % EVAL_INTERVAL)
logger.info('--------------------------------------------------------------------')
