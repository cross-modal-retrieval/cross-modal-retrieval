import os, random, copy
import numpy as np
import torch
import torch.nn as nn
import argparse
import yaml
import shutil
import tensorboard_logger as tb_logger
import logging
import click

import utils
import data
import engine

from vocab import deserialize_vocab



def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/RSITMD_ALAC.yaml', type=str,
                        help='path to a yaml options file')
    parser.add_argument('--resume', default='checkpoint/rsitmd_aba_mv_deno3/1/ALAC_best.pth.tar', type=str,
                        help='path to a yaml options file')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle,Loader=yaml.FullLoader)

    options['optim']['resume'] = opt.resume

    return options

def main(options):
    # choose model
    if options['model']['name'] == "ALAC":
        from layers import ALAC as models
    else:
        raise NotImplementedError
        
    # Create dataset, model, criterion and optimizer
    test_loader = data.get_test_loader(options)

    model = models.factory(options['model'],  
                           cuda=True,
                           data_parallel=False)

    print('Model has {} parameters'.format(utils.params_count(model)))

    # optionally resume from a checkpoint
    if os.path.isfile(options['optim']['resume']):
        print("=> loading checkpoint '{}'".format(options['optim']['resume']))
        checkpoint = torch.load(options['optim']['resume'],map_location=torch.device("cpu"))
        start_epoch = checkpoint['epoch']
        best_rsum = checkpoint['best_rsum']
        model.load_state_dict(checkpoint['model'],strict=False)
    else:
        print("=> no checkpoint found at '{}'".format(options['optim']['resume']))

    # evaluate on test set
    sims = engine.validate_test(test_loader, model)

    return sims

def update_options_savepath(options, k):
    updated_options = copy.deepcopy(options)

    updated_options['optim']['resume'] = options['logs']['ckpt_save_path'] + options['k_fold']['experiment_name'] + "/" \
                                         + str(k) + "/" + options['model']['name'] + '_best.pth.tar'

    return updated_options
    
def i2t_rerank(sim, K1, K2):

    size_i = sim.shape[0]
    size_t = sim.shape[1]
    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)
    sort_i2t_re = np.copy(sort_i2t)[:, :K1]
    address = np.array([])

    for i in range(size_i):
        for j in range(K1):
            result_t = sort_i2t[i][j]
            query = sort_t2i[:, result_t]
            # query = sort_t2i[:K2, result_t]
            address = np.append(address, np.where(query == i)[0][0])

        sort = np.argsort(address)
        sort_i2t_re[i] = sort_i2t_re[i][sort]
        address = np.array([])

    sort_i2t[:,:K1] = sort_i2t_re

    return sort_i2t


def t2i_rerank(sim, K1, K2):

    size_i = sim.shape[0]
    size_t = sim.shape[1]
    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)
    sort_t2i_re = np.copy(sort_t2i)[:K1, :]
    address = np.array([])

    for i in range(size_t):
        for j in range(K1):
            result_i = sort_t2i[j][i]
            query = sort_i2t[result_i, :]
            # query = sort_t2i[:K2, result_t]
            ranks = 1e20
            for k in range(5):
                tmp = np.where(query == i//5 * 5 + k)[0][0]
                if tmp < ranks:
                    ranks = tmp
            address = np.append(address, ranks)

        sort = np.argsort(address)
        sort_t2i_re[:, i] = sort_t2i_re[:, i][sort]
        address = np.array([])

    sort_t2i[:K1, :] = sort_t2i_re

    return sort_t2i

if __name__ == '__main__':
    options = parser_options()

    # run experiment
    one_sims = main(options)
    import mytools
    mytools.save_to_npy(one_sims, "rsitmd_22.npy")
    
    #print(one_sims.shape)
    #rs=np.argsort(one_sims.transpose(),axis=1)+1#
    '''
    # img-text
    c=1
    result=""
    for i in rs[:,-5:]:
        
        result+=str(c)+"-"+str(c+4)+" "+str(i[::-1])+"\n"
        c+=5
    
    
    '''
    '''
    # text-img #need to t()
    c=1
    im=1
    cn=0
    result=""
    for i in rs[:,-5:]:
        cn+=1
        result+=str(c)+"-"+str(im)+" "+str(i[::-1])+"\n"
        c+=1
        if cn%5==0:
           im+=1
        
    
      
    
    f=open("Rsitmd_result_text_img.txt","a+")
    f.write(result)
    '''
    # ave
    last_sims = one_sims

    # get indicators
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(last_sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(last_sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )

    print(all_score)
