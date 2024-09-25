import os,random,copy
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



def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/RSITMD_ALAC.yaml', type=str,
                         help='path to a yaml options file')
    #parser.add_argument('--text_sim_path', default='data/sydney_precomp/train_caps.npy', type=str,help='path to t2t sim matrix')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle,Loader=yaml.FullLoader)

    return options

def main(options):
    # choose model
    
    if options['model']['name'] == "ALAC":
        from layers import ALAC as models
    else:
        raise NotImplementedError
    
    # make ckpt save dir
    if not os.path.exists(options['logs']['ckpt_save_path']):
        os.makedirs(options['logs']['ckpt_save_path'])
    
    # Create dataset, model, criterion and optimizer
    train_loader, val_loader = data.get_loaders(options)
    
    model = models.factory(options['model'],
                           cuda=True, 
                           data_parallel=False)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=options['optim']['lr'])
    
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, [model.pa1, model.pa2]),
                                  lr=10*options['optim']['lr'])
                                  
    print('Model has {} parameters'.format(utils.params_count(model)))

    # optionally resume from a checkpoint
    if options['optim']['resume']:
        if os.path.isfile(options['optim']['resume']):
            print("=> loading checkpoint '{}'".format(options['optim']['resume']))
            checkpoint = torch.load(options['optim']['resume'])
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
         
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
   
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(options['optim']['resume'], start_epoch, best_rsum))
            rsum, all_scores =  engine.validate(val_loader, model)
            print(all_scores)
        else:
            print("=> no checkpoint found at '{}'".format(options['optim']['resume']))
    else:
        start_epoch = 0

    # Train the Model
    best_rsum = 0
    best_score = ""

    for epoch in range(start_epoch, options['optim']['epochs']):

        utils.adjust_learning_rate(options, optimizer, epoch)

        # train for one epoch
        engine.train(train_loader, model, optimizer, epoch, opt=options)

        # evaluate on validation set
        if epoch % options['logs']['eval_step'] == 0:
            rsum, all_scores = engine.validate(val_loader, model, optimizer2)

            is_best = rsum > best_rsum
            if is_best:
                best_score = all_scores
            best_rsum = max(rsum, best_rsum)

            # save ckpt
            utils.save_checkpoint(
                {
                'epoch': epoch + 1,
                'arch': 'baseline',
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'options': options,
                'Eiters': model.Eiters,
            },
                is_best,
                filename='ckpt_{}_{}_{:.2f}.pth.tar'.format(options['model']['name'] ,epoch, best_rsum),
                prefix=options['logs']['ckpt_save_path'],
                model_name=options['model']['name']
            )

            print("Current {}th fold.".format(options['k_fold']['current_num']))
            print("Now  score:")
            print(all_scores)
            print("Best score:")
            print(best_score)

            utils.log_to_txt(
                contexts= "Epoch:{} ".format(epoch+1) + all_scores,
                filename=options['logs']['ckpt_save_path']+ options['model']['name'] + "_" + options['dataset']['datatype'] +".txt"
            )
            utils.log_to_txt(
                contexts= "Best:   " + best_score,
                filename=options['logs']['ckpt_save_path']+ options['model']['name'] + "_" + options['dataset']['datatype'] +".txt"
            )
    
def generate_random_samples(options):
    # load all anns
    auds = utils.load_from_txt(options['dataset']['data_path']+'train_auds.txt')
    fnames = utils.load_from_txt(options['dataset']['data_path']+'train_filename.txt')

    # merge
    assert len(auds) // 5 == len(fnames)
    all_infos = []
    
    for img_id in range(len(fnames)):
        aud_id = [img_id * 5 ,(img_id+1) * 5]
        all_infos.append([auds[aud_id[0]:aud_id[1]], fnames[img_id]])

    # shuffle
    random.shuffle(all_infos)

    # split_trainval
    percent = 0.8
    train_infos = all_infos[:int(len(all_infos)*percent)]
    val_infos = all_infos[int(len(all_infos)*percent):]

    # save to txt
    train_auds = []
    train_fnames = []
    for item in train_infos:
        for aud in item[0]:
            train_auds.append(aud)
        train_fnames.append(item[1])
    utils.log_to_txt(train_auds, options['dataset']['data_path']+'train_auds_verify.txt',mode='w')
    utils.log_to_txt(train_fnames, options['dataset']['data_path']+'train_filename_verify.txt',mode='w')

    val_auds = []
    val_fnames = []
    for item in val_infos:
        for aud in item[0]:
            val_auds.append(aud)
        val_fnames.append(item[1])
    utils.log_to_txt(val_auds, options['dataset']['data_path']+'val_auds_verify.txt',mode='w')
    utils.log_to_txt(val_fnames, options['dataset']['data_path']+'val_filename_verify.txt',mode='w')

    print("Generate random samples to {} complete.".format(options['dataset']['data_path']))

def update_options_savepath(options, k):
    updated_options = copy.deepcopy(options)

    updated_options['k_fold']['current_num'] = k
    updated_options['logs']['ckpt_save_path'] = options['logs']['ckpt_save_path'] + \
                                                options['k_fold']['experiment_name'] + "/" + str(k) + "/"
    return updated_options

if __name__ == '__main__':
    options = parser_options()

    # make logger
    tb_logger.configure(options['logs']['logger_name'], flush_secs=5)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # k_fold verify
    for k in range(options['k_fold']['nums']):
        print("=========================================")
        print("Start {}th fold".format(k))

        # generate random train and val samples
        generate_random_samples(options)

        # update save path
        update_options = update_options_savepath(options, k)

        # run experiment
        main(update_options)
