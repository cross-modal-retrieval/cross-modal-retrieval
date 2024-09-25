from __future__ import print_function

import os
import sys
import time
import json
import pickle
import logging
import argparse

import torch
import numpy as np

import util
import evaluation
import data_provider as data
from common import *
from model import get_model
from bigfile import BigFile
from generic_utils import Progbar


def parse_args():
    parser = argparse.ArgumentParser('W2VVPP predictor')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,
                        help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('testCollection', type=str,
                        help='test collection')
    parser.add_argument('model_path', type=str,
                        help='Path to load the model.')
    parser.add_argument('sim_name', type=str,
                        help='sub-folder where computed similarities are saved')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],
                        help='overwrite existed vocabulary file. (default: 0)')
    parser.add_argument('--query_sets', type=str, default='tv16.avs.txt',
                        help='test query sets,  tv16.avs.txt,tv17.avs.txt,tv18.avs.txt for TRECVID 16/17/18 and tv19.avs.txt for TRECVID19.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='size of a predicting mini-batch.')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of data loader workers.')
 
    args = parser.parse_args()
    return args


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    testCollection = opt.testCollection
    
    resume_file = os.path.join(opt.model_path)
    if not os.path.exists(resume_file):
        logging.info(resume_file + ' not exists.')
        sys.exit(0)

    # Load checkpoint
    logger.info('loading model...')
    checkpoint = torch.load(resume_file)
    epoch = checkpoint['epoch']
    best_perf = checkpoint['best_perf']
    config = checkpoint['config']
    if hasattr(config, 't2v_w2v'):
        w2v_feature_file = os.path.join(rootpath, 'word2vec', 'flickr', 'vec500flickr30m', 'feature.bin')
        config.t2v_w2v.w2v.binary_file = w2v_feature_file

    # Construct the model
    model = get_model('w2vvpp')(config)
    print(model.vis_net)
    print(model.txt_net)

    model.load_state_dict(checkpoint['model'])
    print("=> loaded checkpoint '{}' (epoch {}, best_perf {})"
         .format(resume_file, epoch, best_perf))

    vis_feat_file = BigFile(os.path.join(rootpath, testCollection, 'FeatureData', config.vid_feat))
    vis_ids = map(str.strip, open(os.path.join(rootpath, testCollection, 'VideoSets', testCollection+'.txt')))
    vis_loader = data.vis_provider({'vis_feat': vis_feat_file, 'vis_ids': vis_ids, 'pin_memory': True,
                                    'batch_size': opt.batch_size, 'num_workers': opt.num_workers})

    vis_embs = None

    for query_set in opt.query_sets.split(','):
        output_dir = os.path.join(rootpath, testCollection, 'SimilarityIndex', query_set, opt.sim_name)
        pred_result_file = os.path.join(output_dir, 'id.sent.score.txt')

        if util.checkToSkip(pred_result_file, opt.overwrite):
            continue
        util.makedirs(output_dir)

        if vis_embs is None:
            logger.info('Encoding videos')
            vis_embs, vis_ids = evaluation.encode_vis(model, vis_loader)

        capfile = os.path.join(rootpath, testCollection, 'TextData', query_set)
        # load text data
        txt_loader = data.txt_provider({'capfile': capfile, 'pin_memory': True,
                                    'batch_size': opt.batch_size, 'num_workers': opt.num_workers})

        logger.info('Encoding %s captions' % query_set)
        txt_embs, txt_ids = evaluation.encode_txt(model, txt_loader)

        t2i_matrix = evaluation.compute_sim(txt_embs, vis_embs, measure=config.measure)
        inds = np.argsort(t2i_matrix, axis=1)

        if testCollection == 'msrvtt10ktest':
            label_matrix = np.zeros(inds.shape)
            for index in range(inds.shape[0]):
                ind = inds[index][::-1]
                label_matrix[index][np.where(np.array(vis_ids)[ind]==txt_ids[index].split('#')[0])[0]]=1

            (r1, r5, r10, medr, meanr, mir, mAP) = evaluation.eval(label_matrix)
            sum_recall = r1 + r5 + r10
            tempStr = " * Text to video:\n"
            tempStr += " * r_1_5_10: {}\n".format([round(r1, 3), round(r5, 3), round(r10, 3)])
            tempStr += " * medr, meanr, mir: {}\n".format([round(medr, 3), round(meanr, 3), round(mir, 3)])
            tempStr += " * mAP: {}\n".format(round(mAP, 3))
            tempStr += " * "+'-'*10
            print(tempStr)
            open(os.path.join(output_dir, 'perf.txt'), 'w').write(tempStr)

        start = time.time()
        with open(pred_result_file, 'w') as fout:
            for index in range(inds.shape[0]):
                ind = inds[index][::-1]

                fout.write(txt_ids[index]+' '+' '.join([vis_ids[i]+' %s'%t2i_matrix[index][i]
                    for i in ind])+'\n')
        print('writing result into file time: %.3f seconds\n' % (time.time()-start))



if __name__ == '__main__':
    main()
