import time
import torch
import numpy as np
import sys
from torch.autograd import Variable
import utils
import tensorboard_logger as tb_logger
import logging
from torch.nn.utils.clip_grad import clip_grad_norm

def train(train_loader, model, optimizer, epoch, opt={}):

    # extract value
    grad_clip = opt['optim']['grad_clip']
    max_violation = opt['optim']['max_violation']
    margin = opt['optim']['margin']
    loss_name = opt['model']['name'] + "_" + opt['dataset']['datatype']
    print_freq = opt['logs']['print_freq']

    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()
    

    end = time.time()
    params = list(model.parameters())
    for i, train_data in enumerate(train_loader):
        images, auds, ids= train_data
        
        batch_size = images.size(0)
        margin = float(margin)
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual = Variable(images)
        input_aud = Variable(auds)
        
        
        if torch.cuda.is_available():
            input_visual = input_visual.cuda()
            input_aud = input_aud.cuda()
        
        scores, loss1= model(input_visual, input_aud, True)#
        torch.cuda.synchronize()
        
       
        loss = utils.calcul_loss(scores, input_visual.size(0), margin,max_violation=max_violation, )
        
        
        loss=loss + loss1#+0.5*loss2
        # label_ = torch.eye(batch_size).long()
        # label = Variable(label_).view(-1).cuda()
        # loss_all = criterion(sims.view(-1, 2), label)
        
        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        train_logger.update('L', loss.cpu().data.numpy())


        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        elog=str(train_logger)))

            utils.log_to_txt(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                    .format(epoch, i, len(train_loader),
                            batch_time=batch_time,
                            elog=str(train_logger)),
                opt['logs']['ckpt_save_path']+ opt['model']['name'] + "_" + opt['dataset']['datatype'] +".txt"
            )
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        train_logger.tb_log(tb_logger, step=model.Eiters)

def validate(val_loader, model, optimizer2):

    
    model.train()
    
    total_loss=0
    for i, val_data in enumerate(val_loader):
        images, auds, ids= val_data
          

        input_visual = Variable(images)
        input_aud = Variable(auds)
        
        
        if torch.cuda.is_available():
            input_visual = input_visual.cuda()
            input_aud = input_aud.cuda()
        loss = model(input_visual, input_aud, False, True)
        torch.cuda.synchronize()
        total_loss+=loss.item()
        
        optimizer2.zero_grad()
        loss.backward(retain_graph=True)
        
        
        clip_grad_norm([model.pa1, model.pa2], 2)
        
        torch.cuda.synchronize()
        optimizer2.step()
        torch.cuda.synchronize()
    print('Adaptive Upating Mask Loss:', total_loss)   
    
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 3, 224, 224))
    input_aud = np.zeros((len(val_loader.dataset),1,300,64), dtype=np.float32)

    for i, val_data in enumerate(val_loader):

        images, auds, ids = val_data
        
        for (id, img, aud) in zip(ids, (images.numpy().copy()), (auds.numpy().copy())):
            input_visual[id] = img
            input_aud[id] = aud


    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])
    
    d = utils.shard_dis(input_visual, input_aud, model)

    end = time.time()
    print("calculate similarity time:", end - start)

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(d)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(d)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )
  
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('r1t', r1t, step=model.Eiters)
    tb_logger.log_value('r5t', r5t, step=model.Eiters)
    tb_logger.log_value('r10t', r10t, step=model.Eiters)
    tb_logger.log_value('medrt', medrt, step=model.Eiters)
    tb_logger.log_value('meanrt', meanrt, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore, all_score


def validate_test(val_loader, model):
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 3, 224, 224))
    input_aud = np.zeros((len(val_loader.dataset),1,300,64), dtype=np.float32)
    input_aud_lengeth = [0] * len(val_loader.dataset)
    for i, val_data in enumerate(val_loader):

        images, auds, ids = val_data

        for (id, img, aud) in zip(ids, (images.numpy().copy()), (auds.numpy().copy())):
            input_visual[id] = img
            input_aud[id] = aud
            

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    d = utils.shard_dis(input_visual, input_aud, model)

    end = time.time()
    print("calculate similarity time:", end - start)

    return d
