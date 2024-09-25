import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from GC_attention import ContextBlock
from torch.nn.utils import spectral_norm
from image_model import Resnet18
from res_se_34l_model import ResNetSE34

def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
class Generator(nn.Module):
    def __init__(self, d_conv_dim=32):
        super(Generator, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((3,3))
        self.ContextBlock = ContextBlock(512, 0.25) 
        self.Anc_Prediction = Anc_Prediction()
        self.generator = nn.Sequential(
            nn.Dropout(0.1),
            snlinear(4608, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            snlinear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh()
            )
        self.apply(weights_init)        

    def forward(self, *input):
        Anc_pre, Anc_feat = self.Anc_Prediction(input[0])
        Att1_pre, Att1_feat = self.Anc_Prediction(input[1])
        Att2_pre, Att2_feat = self.Anc_Prediction(input[2])
        Atta2_pre, Anc_A2feat = self.Anc_Prediction(input[3])

        inputv1,inputf1, VF1_Coef = self.ContextBlock(input[0],input[1])
        inputv2,inputf2, VF2_Coef = self.ContextBlock(input[0],input[2])
        
        inputv = 0.5*(inputv1 + inputv2)
        outv1 = forward_feature(self, inputv + Anc_feat)
        outf1 = forward_feature(self, inputf1 + Att1_feat)
        outf2 = forward_feature(self, inputf2 + Att2_feat)

        return outv1, outf1, outf2, Anc_pre, Att1_pre, Att2_pre, Atta2_pre, VF1_Coef, VF2_Coef

    def test_forward(self, *input):
        Anc_pre, Anc_feat = self.Anc_Prediction(input[0])
        Att1_pre, Att1_feat = self.Anc_Prediction(input[1])
        Att2_pre, Att2_feat = self.Anc_Prediction(input[2])

        inputv1,inputf1, VF1_Coef = self.ContextBlock(input[0],input[1])
        inputv2,inputf2, VF2_Coef = self.ContextBlock(input[0],input[2])
        
        inputv = 0.5*(inputv1 + inputv2)
        outv1 = forward_feature(self, inputv + Anc_feat)
        outf1 = forward_feature(self, inputf1 + Att1_feat)
        outf2 = forward_feature(self, inputf2 + Att2_feat)

        return outv1, outf1, outf2

def forward_feature(self,x): 
    N,C,_,_ = x.size()
    x_topk = self.gap(x)
    h3 = self.generator(x_topk.view(N,-1))
    
    return h3


class Dis(nn.Module):
    def __init__(self, d_conv_dim):
        super(Dis, self).__init__()
        self.Dtrans = nn.Sequential(
            nn.Dropout(0.1),
            snlinear(in_features=d_conv_dim*4, out_features=2),
            nn.BatchNorm1d(2),
        )
    def forward(self, in1, in2):
        out1 = self.Dtrans(in1) 
        out2 = self.Dtrans(in2) 
        return out1, out2


class Class(nn.Module):
    def __init__(self, c_conv_dim):
        super(Class, self).__init__()
        self.trans = nn.Sequential(
            nn.Dropout(0.1),
            snlinear(in_features=c_conv_dim*4*2, out_features=2),
            nn.BatchNorm1d(2)
        )

    def forward(self, inp1, inp2, inp3):
        in1 = inp1 - inp2
        in2 = inp1 - inp3
        out = torch.cat([in1, in2], dim=1)
        return self.trans(out)
    def test_forward(self, inp1, inp2, inp3):
        in1 = inp1 - inp2
        in2 = inp1 - inp3
        out = torch.cat([in1, in2], dim=1)
        return self.trans(out)

class Anc_Prediction(nn.Module):
    def __init__(self, c_conv_dim=128, att=6):
        super(Anc_Prediction, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.gap = nn.AdaptiveAvgPool2d((3,3))
        self.encoder = nn.Sequential(
            nn.Dropout(0.1),
            snlinear(4608, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            snlinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )
        self.decoder = nn.Sequential(
            snlinear(128, 512),
            nn.BatchNorm1d(512),
            nn.Sigmoid()
            )
        self.trans = nn.Sequential(
            nn.Dropout(0.1),
            snlinear(in_features=c_conv_dim, out_features=6),
            nn.BatchNorm1d(6)
        )
    def forward(self, anchor):
        B,C,H,W = anchor.size()
        anchor_K = self.gap(anchor)

        En_att = self.encoder(anchor_K.view(B,-1))
        En_att = self.encoder(anchor_K.view(B,-1))
        anc_att = self.trans(En_att)
        anc_att = self.softmax(anc_att)
        att_A = self.softmax(self.decoder(En_att))
        Feat1 = torch.mul(att_A.repeat(1,H*W).reshape(B,C,H,W),anchor)
        Feat1 = torch.mul(anchor,Feat1)
        return anc_att, Feat1



class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        self.feature_audio = ResNetSE34()
        self.feature_frame = Resnet18()

    def forward(self, v1, f1, f2, v2):
        Out_v = self.feature_audio(v1)
        Out_f1 = self.feature_frame(f1)
        Out_f2 = self.feature_frame(f2)
        Out_v2 = self.feature_audio(v2)

        return Out_v, Out_f1, Out_f2, Out_v2

    def test_forward(self, v1, f1, f2):
        Out_v = self.feature_audio(v1)
        Out_f1 = self.feature_frame(f1)
        Out_f2 = self.feature_frame(f2)
        return Out_v, Out_f1, Out_f2