import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from .ALAC_Modules import *
import copy
from layers.crossformer2 import *
from layers.audio import *
from layers.loss import *

import numpy as np
import time



    
class BaseModel(nn.Module):
    def __init__(self, opt={}):
        super(BaseModel, self).__init__()
        
        self.model=CrossFormer(num_classes=13)
        self.model.load_state_dict(torch.load("layers/crossformer-s.pth")["model"],strict=False)

        # aud feature
        self.aud_feature = sresnet18()
        self.aud_feature.load_state_dict(torch.load("layers/audioset_audio_pretrain.pt"),strict=False)
        
        self.cross_attention_s = CrossAttention(opt = opt)

        self.Eiters = 0
        
        self.nt=NTXentLoss(128)
        
        self.pa1 = torch.FloatTensor([1]).cuda()
        self.pa2 = torch.FloatTensor([1]).cuda()
        self.pa1.requires_grad = True
        self.pa2.requires_grad = True
        self.mask = self.pa1 * torch.eye(60).cuda() + self.pa2 * (torch.ones(60).cuda() - torch.eye(60).cuda())
        self.huberloss = nn.SmoothL1Loss(reduce=False)
        self.distill_criterion = nn.MSELoss(reduce=True, size_average=True)
        self.similarity_loss = nn.SmoothL1Loss(reduce=True, size_average=True)
        
        self.similarity_type = 'adapt'
        
        
    def adaptiveUpateMask(self, vid_emb, aud_emb, student_vid_emb, student_aud_emb):
        s1 = cosine_similarity(vid_emb, aud_emb)
        s2 = cosine_similarity(student_vid_emb, student_aud_emb)
        batchsize = self.mask.size(0)
        weight = F.softmax(self.mask,dim=0)
        reweight = torch.ones(batchsize, batchsize).cuda()/(torch.abs(s1).detach() + torch.ones(batchsize, batchsize).cuda()*1e-6)
        weight = reweight * weight
        loss = torch.sum(weight * self.huberloss(s1, s2)) * batchsize
        return loss
        
    def knowlegdeTranfer(self, vid_emb, aud_emb, student_vid_emb, student_aud_emb, *args, **kwargs):
        s1 = cosine_similarity(vid_emb, aud_emb)
        s2 = cosine_similarity(student_vid_emb, student_aud_emb)
        if self.similarity_type=='svd':
            a,b,c = torch.svd(s1)
            s1 = torch.matmul(a,torch.matmul(torch.diag(torch.log(b)), c))
            a,b,c = torch.svd(s2)
            s2 = torch.matmul(a,torch.matmul(torch.diag(torch.log(b)), c))
            loss = self.similarity_loss(s1, s2)
        elif self.similarity_type == 'eig':
            a,b = torch.eig(s1,eigenvectors=True)
            s1 = torch.matmul(b,torch.matmul(torch.diag(a[:,0]), torch.inverse(b)))
            a,b = torch.eig(s2,eigenvectors=True)
            s2 = torch.matmul(b,torch.matmul(torch.diag(a[:,0]), torch.inverse(b)))
            loss = self.similarity_loss(s1, s2)
        elif self.similarity_type == 'diag':
            loss = torch.sum(torch.diagonal(self.huberloss(s1, s2)))
        elif self.similarity_type == 'adapt':
            with torch.no_grad():
                batchsize = self.mask.size(0)
                weight = F.softmax(self.mask,dim=0)
                
            loss = torch.sum(weight.detach() * self.huberloss(s1, s2)) * batchsize
        elif self.similarity_type == 'maxdiag':
            loss = -torch.sum(torch.diagonal(s2))
        else:
            loss = self.similarity_loss(s1, s2)
        
        return loss
        
        
    def forward(self, img, aud, flag=False, val=False):
        
        
        # aud features
        aud_feature = self.aud_feature(aud)
        # img features
        mvsa_feature=self.model(img)
        
        
        if flag==True:
           loss=self.nt(mvsa_feature, aud_feature)
           #loss1=self.distill_criterion(aud_feature, text_feature)
           #loss3=self.nt(mvsa_feature, text_feature)
           #loss1=(1 - cosine_similarity(x1,aud) - 0.8).clamp(min=0).sum()
        
        # VGMF
        Ft,mvsa_feature1 = self.cross_attention_s(mvsa_feature, aud_feature)
        

        # sim dual path 
        #dual_sim=cosine_sim(mvsa_feature,aud_feature)  
        #dual_sim = cosine_similarity(mvsa_feature, text_feature)
        dual_sim = cosine_similarity(mvsa_feature1, Ft)
        
        if flag==True:
           batch_v = mvsa_feature.shape[0]
           batch_t = aud_feature.shape[0]
           stu_img1 = mvsa_feature.unsqueeze(dim=1).expand(-1, batch_t, -1)
           stu_aud1 = aud_feature.unsqueeze(dim=0).expand(batch_v, -1, -1)
           loss2 = self.knowlegdeTranfer(mvsa_feature1.detach(), Ft.detach(), stu_img1, stu_aud1)
        elif flag==False and val==True:
           batch_v = mvsa_feature.shape[0]
           batch_t = aud_feature.shape[0]
           stu_img1 = mvsa_feature.unsqueeze(dim=1).expand(-1, batch_t, -1)
           stu_aud1 = aud_feature.unsqueeze(dim=0).expand(batch_v, -1, -1)
           loss2 = self.adaptiveUpateMask(mvsa_feature1.detach(), Ft.detach(), stu_img1.detach(), stu_aud1.detach())
        
        if flag==False and val==True:
           return loss2
           
        if flag==False:
           return dual_sim
        return dual_sim,0.1*loss+0.1*loss2
        
        


def factory(opt, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = BaseModel(opt)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model
