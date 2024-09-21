import os
import numpy as np
import time
import scipy.io as scio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import dset
from models import ImgNet, TxtNet
from metric import compress, calculate_map
import settings


def save_hash_code(query_text, query_image, query_label, retrieval_text, retrieval_image, retrieval_label, dataname, bit):
    if not os.path.exists('./Hashcode'):
        os.makedirs('./Hashcode')

    save_path = './Hashcode/' + dataname + '_' + str(bit) + 'bits.mat'
    scio.savemat(save_path, 
                 {'query_text': query_text,
                 'query_image': query_image,
                 'query_label': query_label, 
                 'retrieval_text': retrieval_text, 
                 'retrieval_image': retrieval_image, 
                 'retrieval_label': retrieval_label})

class Session:
    def __init__(self):
        self.logger = settings.logger

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        # torch.cuda.set_device(settings.GPU_ID)

        self.train_dataset = dset.MY_DATASET(train=True, transform=dset.train_transform)
        self.test_dataset = dset.MY_DATASET(train=False, database=False, transform=dset.test_transform)
        self.database_dataset = dset.MY_DATASET(train=False, database=True, transform=dset.test_transform)

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=settings.BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=settings.NUM_WORKERS,
                                                   drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=settings.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=settings.NUM_WORKERS)

        self.database_loader = torch.utils.data.DataLoader(dataset=self.database_dataset,
                                                  batch_size=settings.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=settings.NUM_WORKERS)

        self.CodeNet_I = ImgNet(code_len=settings.CODE_LEN)
        self.FeatNet_I = ImgNet(code_len=settings.CODE_LEN)

        txt_feat_len = dset.txt_feat_len
        self.CodeNet_T = TxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)

        self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

    def train(self, epoch):
        self.CodeNet_I.cuda().train()
        self.FeatNet_I.cuda().eval()
        self.CodeNet_T.cuda().train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)
        
        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (epoch + 1, settings.NUM_EPOCH, self.CodeNet_I.alpha, self.CodeNet_T.alpha))

        for idx, (img, F_T, labels, _) in enumerate(self.train_loader):
            img = Variable(img.cuda())
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())
            labels = Variable(labels.cuda())


            self.opt_I.zero_grad()
            self.opt_T.zero_grad()

            F_I , _, _ = self.FeatNet_I(img)
            _, hid_I, code_I = self.CodeNet_I(img)
            _, hid_T, code_T = self.CodeNet_T(F_T)


            F_I = F.normalize(F_I)
            S_I = F_I.mm(F_I.t())
            S_I = S_I * 2 - 1

            F_T = F.normalize(F_T)
            S_T = F_T.mm(F_T.t())
            S_T = S_T * 2 - 1

            
            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)
            
            BI_BI = B_I.mm(B_I.t())
            BT_BT = B_T.mm(B_T.t())
            BI_BT = B_I.mm(B_T.t())

            S_tilde = settings.BETA * S_I + (1 - settings.BETA) * S_T
            S = (1 - settings.ETA) * S_tilde + settings.ETA * S_tilde.mm(S_tilde) / settings.BATCH_SIZE
            S = S * settings.MU

            loss1 = F.mse_loss(BI_BI, S)
            loss2 = F.mse_loss(BI_BT, S)
            loss3 = F.mse_loss(BT_BT, S)
            loss = settings.LAMBDA1 * loss1 + 1 * loss2 + settings.LAMBDA2 * loss3

            loss.backward()
            self.opt_I.step()
            self.opt_T.step()

            if (idx + 1) % (len(self.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f Loss3: %.4f Total Loss: %.4f'
                    % (epoch + 1, settings.NUM_EPOCH, idx + 1, len(self.train_dataset) // settings.BATCH_SIZE,
                        loss1.item(), loss2.item(), loss3.item(), loss.item()))

    def eval(self):
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
        
        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval().cuda()  
        self.CodeNet_T.eval().cuda()

        t1 = time.time()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I, self.CodeNet_T, self.database_dataset, self.test_dataset)
          
        MAP_I2T = calculate_map(qu_BI, re_BT, qu_L, re_L)
        MAP_T2I = calculate_map(qu_BT, re_BI, qu_L, re_L)
        t2 = time.time()

        print('[%d]Test time:%.2f ...test map: map(i->t): %3.4f, map(t->i): %3.4f' % (settings.CODE_LEN, (t2 - t1), MAP_I2T, MAP_T2I))
        
        if settings.FLAG_savecode:
            save_hash_code(qu_BT, qu_BI, qu_L, re_BT, re_BI, re_L, settings.DATASET_NAME, settings.CODE_LEN)

        with open('result/' + settings.DATASET_NAME + '.txt', 'a+') as f:
            f.write('[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (settings.DATASET_NAME, settings.CODE_LEN, MAP_I2T, MAP_T2I))

        self.logger.info('--------------------------------------------------------------------')

    def save_checkpoints(self, step, file_name='latest.pth'):
        ckp_path = os.path.join(settings.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')
 
    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = os.path.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])


def main():
    
    sess = Session()

    if settings.EVAL == True:
        sess.load_checkpoints()
        sess.eval()

    else:
        start_time = time.time()
        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            sess.train(epoch)
            # eval the Model
            # if (epoch + 1) % settings.EVAL_INTERVAL == 0:
            #     sess.eval()
            # save the model
            # if epoch + 1 == settings.NUM_EPOCH:
            #     sess.save_checkpoints(step=epoch+1)   
        train_time = time.time() - start_time
        print('Training time: ', train_time)
        sess.eval()


if __name__ == '__main__':
    if not os.path.exists('result'):
        os.makedirs('result')
    if not os.path.exists('log'):
        os.makedirs('log')
        
    main()
