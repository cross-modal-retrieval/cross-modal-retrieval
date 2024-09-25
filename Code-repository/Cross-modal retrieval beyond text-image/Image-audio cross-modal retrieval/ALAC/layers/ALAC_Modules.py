import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import math
import copy

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_similarity(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
    
def sim(img,text):
    w12=img*text
    w1=torch.norm(img,2)
    w2=torch.norm(text,2)
    return w12/(w1*w2)

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    
    im = l2norm(im, dim=-1)
    s = l2norm(s, dim=-1)
    
    w12 = im.mm(s.t())
    return w12  
    
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = (torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()) + eps
    
    X = torch.div(X, norm)
    return X

# cross attention
class CrossAttention(nn.Module):

    def __init__(self, opt={}):
        super(CrossAttention, self).__init__()

        self.att_type = opt['cross_attention']['att_type']
        
        dim = opt['embed']['embed_dim']
        
        self.softmax=nn.Softmax(-1)

        if self.att_type == "soft_att":
            self.cross_attention = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
        elif self.att_type == "fusion_att":
            self.cross_attention_fc1 = nn.Sequential(
                nn.Linear(2*dim, dim),
                nn.Sigmoid()
            )
            self.cross_attention_fc2 = nn.Sequential(
                nn.Linear(2*dim, dim),
            )
            self.cross_attention = lambda x:self.cross_attention_fc1(x)*self.cross_attention_fc2(x)

        elif self.att_type == "similarity_att":
            self.fc_visual = nn.Sequential(
                nn.Linear(dim, dim),
            )
            self.fc_text = nn.Sequential(
                nn.Linear(dim, dim),
            )
        elif self.att_type == "sim_att":
            self.fc_visual = nn.Sequential(
                nn.Linear(dim, dim),
            )
            self.fc_text = nn.Sequential(
                nn.Linear(dim, dim),
            )
        elif self.att_type == "ls_att":
            self.fc_visual = nn.Sequential(
                nn.Linear(dim, dim),
            )
            self.fc_text = nn.Sequential(
                nn.Linear(dim, dim),
            ) 
            #self.cross_attention = lambda x:self.fc_text(x)*self.fc_visual(x)
             
        elif self.att_type == "ca_att":
            self.ca=CA()
        elif self.att_type == "ha_att":
            self.ha=ha()
        elif self.att_type == "sa_att":
            self.sa=sa()
        else:
            raise Exception

    def forward(self, visual, text):
        batch_v = visual.shape[0]
        batch_t = text.shape[0]

        if self.att_type == "soft_att":
            visual_gate = self.cross_attention(visual)

            # mm
            visual_gate = visual_gate.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            return visual_gate*text

        elif self.att_type == "fusion_att":
            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            fusion_vec = torch.cat([visual,text], dim=-1)

            return self.cross_attention(fusion_vec)
        elif self.att_type == "similarity_att":
            visual = self.fc_visual(visual)
            text = self.fc_text(text)

            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)
            
            sims = visual*text
            return F.sigmoid(sims) * text
        elif self.att_type == "ha_att":
            

            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)
            
            
            return self.ha(visual,text)
        elif self.att_type == "sim_att":
            visual = self.fc_visual(visual)

            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)
            visual=sim(visual,text)
            #visual=F.relu(visual)/torch.norm(visual,2)
            return torch.sigmoid(visual)*text
        elif self.att_type=="ls_att":
            #visual = self.fc_visual(visual)
            #text = self.fc_text(text)
            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)
            
            text=torch.sigmoid(visual*text)*text
            
            visual=torch.sigmoid(visual*text)*visual
            return text,visual
        elif self.att_type=="ca_att":
            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)
            
            text = self.ca(visual,text)+text
            
            #text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)
            
            return text
        elif self.att_type=="sa_att":
            visual,text=self.sa(visual,text)
            
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)
            
            
            return visual,text
 
