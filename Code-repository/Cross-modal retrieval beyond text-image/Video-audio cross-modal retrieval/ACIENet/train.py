import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
from utils import *
import warnings
from model_source import Generator, Dis, Class, feature_extractor
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import numpy as np
import math
import samplers


def load(feature, D, G, C):
    states = torch.load('V2F20_0.9720207253886011.pkl')
    feature.load_state_dict(states['feature'])
    G.load_state_dict(states['G'])
    D.load_state_dict(states['D'])
    C.load_state_dict(states['C'])
    return feature, D, G, C

def adjust_lr(optimizer,optimizer2, epoch):
    if 20 == epoch :
        for p1 in optimizer.param_groups:
            p1['lr'] = p1['lr'] * 0.1
        for p2 in optimizer2.param_groups:
            p2['lr'] = p2['lr'] * 0.1
    if 20 < epoch < 60 and epoch % 35 == 0:
        for p1 in optimizer.param_groups:
            p1['lr'] = p1['lr'] * 0.1
        for p2 in optimizer2.param_groups:
            p2['lr'] = p2['lr'] * 0.1
            
def positive_F(labels, f,*a):
    positive = torch.zeros_like(f).to('cuda')
    for i in range(0,len(a)):
        positive_B = torch.zeros_like(f)
        positive_B [labels == i]=1
        positive += torch.mul(positive_B,a[i])
    return positive    

def train():
    d_conv_dim=32
    g_conv_dim=32
    c_conv_dim=32
    feature = feature_extractor()
    generator= Generator(g_conv_dim)
    Discri = Dis(d_conv_dim)
    Cls = Class(c_conv_dim)

    Rank_loss = lift_4struct(1.2,2) 
    Loss_Fc = nn.CrossEntropyLoss() 
    cuda = True if torch.cuda.is_available() else False
    #feature, Discri, generator, Cls = load(feature, Discri, generator, Cls)
    if cuda:
        feature = feature.to('cuda')
        generator = generator.to('cuda')
        Discri = Discri.to('cuda')
        Cls = Cls.to('cuda')
        Rank_loss = Rank_loss.to('cuda')
        Loss_Fc = Loss_Fc.to('cuda')
    acc_best = 0
    sampler_R = 'RandomIdentitySampler'
    MyDataSet = train_data_V2F()
    faceroot= './train_V2F_8vox1_label50.txt'
    train_dataset=[]
    with open(faceroot, 'r') as f:
        for line in f:
            strr = line.split()
            item = {
                "voice_path": strr[0],
                "image_path1": strr[1],
                "image_path2": strr[2],
                "id": strr[3]
            }
            train_dataset.append(item)

    sampler = getattr(samplers, sampler_R)(train_dataset, batch_size=50, num_instances=50)
    dataloader = DataLoader(MyDataSet, sampler=sampler, batch_size=50,
                              num_workers=4, pin_memory=True)

    optimizer1 = torch.optim.Adam(
        [
            {"params": feature.parameters(), "lr": 5e-2},
            {"params": generator.parameters(), "lr": 5e-3},
            {"params": Cls.parameters(), "lr": 5e-2},
            
        ],
    )
    optimizer2 = torch.optim.Adam(Discri.parameters(), lr=5e-3, betas=(0.5, 0.999))
    for epoch in range(50):
        adjust_lr(optimizer1,optimizer2, epoch)
        feature.train()
        generator.train()
        Discri.train()
        Cls.train()
        count_train = 0.0   
        audio_count = 0.0   
        face_count = 0.0 
        total_train = 0.0  
        audio_Att = 0.0  
        face_Att = 0.0 
        for i, data in enumerate(dataloader):
            a, f1, f2, a2, ID, label, face_m, audio_m, Att_a, Att_f1, Att_f2, Att_a2 = data
            a = a.to('cuda')
            f1 = f1.to('cuda')
            f2 = f2.to('cuda')
            a2 = a2.to('cuda')
            Att_a, Att_f1, Att_f2, Att_a2 = Att_a.to('cuda'), Att_f1.to('cuda'), Att_f2.to('cuda'), Att_a2.to('cuda')
            label1=label
            label1 = label1.to('cuda')
            total_train += a.size(0)
            face_m = face_m.to('cuda')
            audio_m = audio_m.to('cuda')
            a, f1, f2, a2 = feature(a, f1, f2, a2)
            label_0 = torch.zeros(label1.size()).cuda()
            label_1 = torch.ones(label1.size()).cuda()
            a, f1, f2, Anc_pre, Att1_pre, Att2_pre, Atta2_pre, VF1_Coef, VF2_Coef = generator(a, f1, f2, a2,'train')
            mask0 = (label1 == 0).to(dtype=torch.int32)
            mask1 = (label1 == 1).to(dtype=torch.int32)
            Coef1_loss = torch.pow(torch.sigmoid(5*VF1_Coef)-mask0, 2).sum([0,1], keepdim=True).clamp(min=1e-12).sqrt()
            Coef2_loss = torch.pow(torch.sigmoid(5*VF2_Coef)-mask1, 2).sum([0,1], keepdim=True).clamp(min=1e-12).sqrt()
            coff = epoch/50*math.pi*torch.ones(1).to('cuda') 
            Coef_total_loss = 0.5*(1+torch.cos(coff))*(Coef1_loss + Coef2_loss)/(2*a.size(0))

            #######################################
            for p1 in Discri.parameters():
                p1.requires_grad = True
            for p2 in feature.parameters():
                p2.requires_grad = False
            for p3 in generator.parameters():
                p3.requires_grad = False
            for p4 in Cls.parameters():
                p4.requires_grad = False
        
            FP = positive_F(label1, a, f1, f2)
            out1, out2 = Discri(a, FP)
            loss_d = Loss_Fc(out1, audio_m) + Loss_Fc(out2, face_m)

            loss_d_total = loss_d
            optimizer2.zero_grad()
            loss_d_total.backward(retain_graph=True)
            for p in Discri.parameters():
                torch.nn.utils.clip_grad_norm(p.data, 0.02)
            optimizer2.step()
            audio_count += label_acc(out1, audio_m)
            face_count += label_acc(out2, face_m)
            #######################################
            for p1 in Discri.parameters():
                p1.requires_grad = False
            for p2 in feature.parameters():
                p2.requires_grad = True
            for p3 in generator.parameters():
                p3.requires_grad = True
            for p4 in Cls.parameters():
                p4.requires_grad = True

            out1, out2 = Discri(a, FP)
            loss1_g = Loss_Fc(out1, face_m) + Loss_Fc(out2, audio_m)

            predict0 = Cls(a, f1, f2)
            loss_p = Loss_Fc(predict0, label1)

            loss_Aatt = Loss_Fc(Anc_pre, Att_a) + Loss_Fc(Atta2_pre, Att_a2)
            loss_Fatt = Loss_Fc(Att1_pre, Att_f1) + Loss_Fc(Att2_pre, Att_f2)

            loss_m = compute_metric(label, Rank_loss, a, f1, f2)
            
            loss_total = loss1_g + 2*loss_m + 2*loss_p + 1*Coef_total_loss + loss_Aatt + loss_Fatt
            audio_Att += 2*label_acc(Anc_pre, Att_a)
            face_Att += label_acc(Att1_pre, Att_f1) + label_acc(Att2_pre, Att_f2)
            
            count_train += label_acc(predict0, label)
            if i % 10 == 0:
                print(epoch, i, 'G ', loss1_g.item(), ' M ', loss_m.item(), ' C ', loss_p.item(), 'D ', loss_d.item(), 'coef', Coef_total_loss.item())
                print('A_att', loss_Aatt.item(), ' F_att ', loss_Fatt.item())
                if count_train != 0:
                    print('counts =', count_train)

            optimizer1.zero_grad()
            loss_total.backward()
            for p3 in generator.parameters():
                torch.nn.utils.clip_grad_norm(p3.data, 0.02)
            optimizer1.step()

        audio_acc = audio_count / total_train
        face_acc = face_count / (total_train)
        acc = count_train / (total_train )

        audio_Att_acc = audio_Att/ (total_train * 2)
        face_Att_acc = face_Att / (total_train * 2)

        print('epoch:', epoch, 'V2F training acc :', acc)
        print('Audio acc : ', audio_acc, 'Face acc : ', face_acc)
        print('Audio att_acc : ', audio_Att_acc, 'Face att_acc : ', face_Att_acc)
        acc_best = eval(feature, generator, Cls, Discri, epoch, acc_best)

    print('training over')


if __name__ == '__main__':
    seed = 25
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train()
