import torch
import numpy as np
import sys
import  math
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn
import shutil
import torch.nn.functional as F

# 保存结果到txt文件
def log_to_txt( contexts=None,filename="save.txt", mark=False,encoding='UTF-8',mode='a'):
    f = open(filename, mode,encoding=encoding)
    if mark:
        sig = "------------------------------------------------\n"
        f.write(sig)
    elif isinstance(contexts, dict):
        tmp = ""
        for c in contexts.keys():
            tmp += str(c)+" | "+ str(contexts[c]) +"\n"
        contexts = tmp
        f.write(contexts)
    else:
        if isinstance(contexts,list):
            tmp = ""
            for c in contexts:
                tmp += str(c)
            contexts = tmp
        else:
            contexts = contexts + "\n"
        f.write(contexts)


    f.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)

def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]
    return dict_to

def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count


def collect_match(input):
    """change the model output to the match matrix"""
    image_size = input.size(0)
    text_size = input.size(1)

    # match_v = torch.zeros(image_size, text_size, 1)
    # match_v = match_v.view(image_size*text_size, 1)
    input_ = nn.LogSoftmax(2)(input)
    output = torch.index_select(input_, 2, Variable(torch.LongTensor([1])).cuda())

    return output


def collect_neg(input):
    """"collect the hard negative sample"""
    if input.dim() != 2:
        return ValueError

    batch_size = input.size(0)
    mask = Variable(torch.eye(batch_size)>0.5).cuda()
    output = input.masked_fill_(mask, 0)
    output_r = output.max(1)[0]
    output_c = output.max(0)[0]
    loss_n = torch.mean(output_r) + torch.mean(output_c)
    return loss_n
    
def ct(score,margin):
    a=-torch.exp(5*score)+torch.exp(torch.Tensor([5]).cuda())
    b=torch.exp(torch.Tensor([5]).cuda())-1
    c=(a/b)*margin
    return c


def calcul_loss(scores, size, margin,loss_type="mse",max_violation=False, text_sim_matrix=None, param = "0.8 | 5"):

    diagonal = scores.diag().view(size, 1)
    #margin=ct(torch.sigmoid(scores),margin)
    
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    
    
    # compare every diagonal score to scores in its column
    # caption retrieval img--text
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval   text--img
    cost_im = (margin + scores - d2).clamp(min=0)

    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)
    
    

    if max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
        
       
            
        
    #cost_s=((cost_s.pow(2))).sum(1)
    
    #cost_s[cost_s<=0]=1e-8
    #cost_s=cost_s.pow(1/2)
    #cost_im=((cost_im.pow(2))).sum(0)
    
    #cost_im[cost_im<=0]=1e-8
    #cost_im=cost_im.pow(1/2)
    
    return cost_s.sum() + cost_im.sum()

from torch.autograd import Variable
def bce(scores):
    
    eps = 0.000001

    scores = scores.clamp(min=eps, max=(1.0-eps))
    de_scores = 0.2 - scores

    label = Variable(torch.eye(scores.size(0))).cuda()
    de_label = 1 - label
        
    scores = torch.log(scores.pow(2)) * label
    de_scores = torch.log(de_scores.pow(2)) * de_label

    if True:
        le = -(scores.sum() + scores.sum() + de_scores.min(1)[0].sum() + de_scores.min(0)[0].sum())
    else:
        le = -(scores.diag().mean() + de_scores.mean())

    return le
    
def compute_loss(scores, batch_hard_count=0):
    
    scores=2-2*scores
    diagonal = scores.diag().view(scores.size(0), 1)
    #margin=ct(torch.sigmoid(scores),margin)
    
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    loss_weight=10
    if batch_hard_count == 0:
        pair_n = scores.size(0) * (scores.size(0) - 1.0)

        # compare every diagonal score to scores in its column
        # caption retrieval img--text
        cost_s = (d1 - scores)
        # compare every diagonal score to scores in its row
        # image retrieval   text--img
        cost_im = (d2 - scores)

        
        cost_s=torch.sum(torch.log(1+torch.exp(cost_s*loss_weight))) / pair_n
        cost_im=torch.sum(torch.log(1+torch.exp(cost_im*loss_weight)))/ pair_n
    
        

    else:
        # compare every diagonal score to scores in its column
        # caption retrieval img--text
        cost_s = (d1 - scores)
        # compare every diagonal score to scores in its row
        # image retrieval   text--img
        cost_im = (d2 - scores)

        cost_s=torch.log(1+torch.exp(cost_s*loss_weight))
        cost_s, _ = torch.sort(cost_s, dim=1, descending=False)
        cost_s,_= torch.topk(cost_s, batch_hard_count)
        cost_im=torch.log(1+torch.exp(cost_im*loss_weight))
        cost_im, _ = torch.sort(cost_im, dim=1, descending=False)
        cost_im,_= torch.topk(cost_im, batch_hard_count)
        
        cost_s=cost_s.mean()
        cost_im=cost_im.mean()
        
        
    
    return (cost_s + cost_im)/2
    
class NceLoss(nn.Module):
    # 初始化batch size以及top k hard sample参数
    def __init__(self, top_k=2, scale=100.0):
        super(NceLoss, self).__init__()
        self.top_k = top_k
        self.scale = scale
        self.mulcls=nn.CrossEntropyLoss()

    def forward(self, scores,size):
        # 计算图像-文本的分数矩阵

        diag = scores.diag()
        mask = torch.eye(scores.size(0)) > 0.5
        if torch.cuda.is_available():
            mask = mask.cuda()

        targets = torch.LongTensor(np.zeros(size)).cuda()

        scores = scores.masked_fill_(mask, 3.0)
        # 分数由高到低排序，并取topK结果
        s_i2t, _ = torch.sort(scores, dim=1, descending=True)
        s_i2t= s_i2t[:, :self.top_k]
        s_i2t[:, 0] = diag
        s_t2i, _ = torch.sort(scores.t(), dim=1, descending=True)
        s_t2i = s_t2i[:, :self.top_k]
        s_t2i[:,0] = diag
        s_i2t = self.scale * s_i2t
        s_t2i = self.scale * s_t2i
        return self.mulcls(s_i2t, targets) + self.mulcls(s_t2i, targets)

class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=1, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = torch.nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()

def acc_train(input):
    predicted = input.squeeze().numpy()
    batch_size = predicted.shape[0]
    predicted[predicted > math.log(0.5)] = 1
    predicted[predicted < math.log(0.5)] = 0
    target = np.eye(batch_size)
    recall = np.sum(predicted * target) / np.sum(target)
    precision = np.sum(predicted * target) / np.sum(predicted)
    acc = 1 - np.sum(abs(predicted - target)) / (target.shape[0] * target.shape[1])

    return acc, recall, precision

def acc_i2t(input):
    """Computes the precision@k for the specified values of k of i2t"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    # ranks_ = np.zeros(image_size//5)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = np.argsort(input[index])[::-1]
        # Score
        rank = 1e20
        # index_ = index // 5
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]

            if tmp < rank:
                rank = tmp
        if rank == 1e20:
            print('error')
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i(input):
    """Computes the precision@k for the specified values of k of t2i"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)
    # ranks_ = np.zeros(image_size // 5)
    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = np.argsort(input[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)

def shard_dis(images, auds, model, shard_size=112):
    """compute image-caption pairwise distance during validation and test"""

    n_im_shard = (len(images) - 1) // shard_size + 1
    n_aud_shard = (len(auds) - 1) // shard_size + 1

    d = np.zeros((len(images), len(auds)))

    for i in range(n_im_shard):
        
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        #print(im_start,im_end,im_start-im_end)
#        print("======================")
#        print("im_start:",im_start)
#        print("im_end:",im_end)

        for j in range(n_aud_shard):
            
            sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (i,j))
            aud_start, aud_end = shard_size * j, min(shard_size * (j + 1), len(auds))
            
            
            with torch.no_grad():
                 im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).float().cuda()
                 a = Variable(torch.from_numpy(auds[aud_start:aud_end]), volatile=True).cuda()
                 

                 sim = model(im, a)
                 sim = sim.squeeze()
                 d[im_start:im_end, aud_start:aud_end] = sim.data.cpu().numpy()
    print("----------------------")
    sys.stdout.write('\n')
    return d

def acc_i2t2(input):
    """Computes the precision@k for the specified values of k of i2t"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = np.argsort(input[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i2(input):
    """Computes the precision@k for the specified values of k of t2i"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)

    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = np.argsort(input[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)

def shard_dis_reg(images, captions, model, shard_size=128, lengths=None):
    """compute image-caption pairwise distance during validation and test"""

    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))

    for i in range(len(images)):
        # im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        im_index = i
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))

            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            im = Variable(torch.from_numpy(images[i]), volatile=True).float().unsqueeze(0).expand(len(s), 3, 256, 256).cuda()

            l = lengths[cap_start:cap_end]

            sim = model(im, s, l)[:, 1]



            sim = sim.squeeze()
            d[i, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def save_checkpoint(state, is_best, filename, prefix='', model_name = None):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            # torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix +model_name +'_best.pth.tar')

        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

def adjust_learning_rate(options, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        lr = param_group['lr']

        if epoch % options['optim']['lr_update_epoch'] == options['optim']['lr_update_epoch'] - 1:
            lr = lr * options['optim']['lr_decay_param']

        param_group['lr'] = lr

    print("Current lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

def load_from_txt(filename, encoding="utf-8"):
    f = open(filename,'r' ,encoding=encoding)
    contexts = f.readlines()
    return contexts
