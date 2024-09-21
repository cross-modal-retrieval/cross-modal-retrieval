from JDSH import JDSH
from utils import logger
from args import config
import time
import os

def log_info(logger, config):

    logger.info('--- Configs List---')
    logger.info('--- Dataset name:{}'.format(config.DATASET_NAME))
    logger.info('--- Bit:{}'.format(config.HASH_BIT))
    logger.info('--- Train:{}'.format(config.TRAIN))
    logger.info('--- Alpha:{}'.format(config.alpha))
    logger.info('--- Beta:{}'.format(config.beta))
    logger.info('--- Lambda:{}'.format(config.lamb))
    logger.info('--- Mu:{}'.format(config.mu))
    logger.info('--- Batch:{}'.format(config.BATCH_SIZE))
    logger.info('--- Lr_IMG:{}'.format(config.LR_IMG))
    logger.info('--- Lr_TXT:{}'.format(config.LR_TXT))


def main():
    print('\n\n\n\n\n\n\n\n\n')
    # log
    log = logger()
    log_info(log, config)

    Model = JDSH(log, config)

    if config.TRAIN == False:
        Model.load_checkpoints(config.CHECKPOINT)
        Model.eval()

    else:
        t1 = time.time()
        for epoch in range(config.NUM_EPOCH):
            Model.train(epoch)
            '''
            if (epoch + 1) % config.EVAL_INTERVAL == 0:
                Model.eval()
            # save the model
            if epoch + 1 == config.NUM_EPOCH:
                Model.save_checkpoints()
            '''
        t2 = time.time()
        print('Training time %.3f' % (t2 - t1))
        Model.eval()

if __name__ == '__main__':
    if not os.path.exists('result'):
        os.makedirs('result')
    if not os.path.exists('logs'):
        os.makedirs('logs')

    main()
