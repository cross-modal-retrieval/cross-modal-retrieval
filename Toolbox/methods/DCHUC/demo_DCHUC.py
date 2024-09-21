import pickle
import os
import argparse
import torch
import time
import numpy as np
import torch.optim as optim
import scipy.io as scio
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.data_processing as dp
import utils.dchuc_loss as dl
import utils.cnn_model as cnn_model
import utils.txt_module as txt_model
import utils.subset_sampler as subsetsampler
import utils.calc_hr as calc_hr

parser = argparse.ArgumentParser(description="DCHUC demo")
parser.add_argument('--bits', default='64', type=int,
                    help='binary code length (default: 32,48,64)') 
# parser.add_argument('--gpu', default='3', type=str,
#                     help='selected gpu (default: 3)')
parser.add_argument('--arch', default='alexnet', type=str,
                    help='model name (default: alexnet)')
parser.add_argument('--max-iter', default=30, type=int,
                    help='maximum iteration (default: 30)')
parser.add_argument('--epochs', default=3, type=int,
                    help='number of epochs (default: 3)')
parser.add_argument('--batch-size', default=64, type=int,
                    help='batch size (default: 64)')

parser.add_argument('--num-samples', default=2000, type=int,
                    help='hyper-parameter: number of samples (default: 2000)')
parser.add_argument('--gamma', default=200, type=int,
                    help='hyper-parameter: gamma (default: 200)')
parser.add_argument('--beta', default=1, type=int,
                    help='hyper-parameter: beta (default: 1)')
parser.add_argument('--alpha', default=50, type=int,
                    help='hyper-parameter: alpha (default: 50)')
parser.add_argument('--yita', default=50, type=int,
                    help='hyper-parameter: yita (default: 50)')
parser.add_argument('--mu', default=50, type=int,
                    help='hyper-parameter: mu (default: 50)')
parser.add_argument('--learning-rate', default=0.0001, type=float,
                    help='hyper-parameter: learning rate (default: 0.0001)')
parser.add_argument('--learning-rate-txt', default=0.004, type=float,
                    help='hyper-parameter: learning rate (default: 0.004)')
parser.add_argument('--y_dim', default=1000, type=int,
                    help='txt dim (default: 1000)') # TODO
parser.add_argument('--dataname', default='flickr', type=str,
                    help='flickr/nuswide/coco') # TODO
parser.add_argument('--FLAG_savecode', default=0, type=int,
                    help='save the hash codes') 
opt = parser.parse_args()

def save_hash_code(query_text, query_image, query_label, retrieval_text, retrieval_image, retrieval_label, dataname, bit):
    if not os.path.exists('./Hashcode'):
        os.makedirs('./Hashcode')

    save_path = './Hashcode/'+ dataname + '_' + str(bit) + 'bits.mat'
    scio.savemat(save_path, 
                 {'query_text': query_text,
                  'query_image': query_image,
                  'query_label': query_label, 
                  'retrieval_text': retrieval_text, 
                  'retrieval_image': retrieval_image, 
                  'retrieval_label': retrieval_label})

def _dataset(dataname):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dset_database = dp.DatasetProcessing(
        dataname, 'retrieval', transformations
    )
    dset_database_txt = dp.DatasetProcessing_txt(
        dataname, 'retrieval'
    )
    dset_test = dp.DatasetProcessing(
        dataname, 'query', transformations
    )
    dset_test_txt = dp.DatasetProcessing_txt(
        dataname, 'query'
    )
    dset_train = dp.DatasetProcessing(
        dataname, 'train', transformations
    )
    dset_train_txt = dp.DatasetProcessing_txt(
        dataname, 'train'
    )
    num_database, num_test, num_train = len(dset_database), len(dset_test), len(dset_train)


    databaselabels, testlabels, trainlabels = torch.from_numpy(dset_database.label), torch.from_numpy(dset_test.label), torch.from_numpy(dset_train.label)

    dsets = (dset_database, dset_database_txt, dset_test, dset_test_txt, dset_train, dset_train_txt)
    nums = (num_database, num_test, num_train)
    labels = (databaselabels, testlabels, trainlabels)
    return nums, dsets, labels

def calc_sim(database_label, train_label):
    S = (database_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    r = S.sum() / (1-S).sum()
    S = S*(1+r) - r
    return S

def calc_loss(W, L, V, U, G, S, S_1, code_length, select_index, gamma, yita, alpha, mu):
    num_database = V.shape[0]
    square_loss = (U.dot(V.transpose()) - code_length*S) ** 2 + (G.dot(V.transpose()) - code_length*S) ** 2

    square_loss_1 = (U.dot((G.transpose())) - code_length * S_1) **2
    V_omega = V[select_index, :]
    quantization_loss = (0.5 * (U + G) - V_omega) ** 2
    label_loss = (U.dot(W) - L.numpy())
    l2 = W * W
    loss = (square_loss.sum() + gamma * quantization_loss.sum() + alpha * label_loss.sum() + mu * square_loss_1.sum()
            + yita * l2.sum()) / (opt.num_samples * num_database)
    return loss

def encode(model, data_loader, num_data, bit, istxt=False):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        if istxt:
            data_input = data_input.unsqueeze(1).unsqueeze(-1).type(torch.FloatTensor)
        data_input = Variable(data_input.cuda())
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
    return B

def adjusting_learning_rate(optimizer, iter, text=False):
    if text:
        update_list = [10, 20, 36]
        if iter in update_list:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 1.5
    else:
        update_list = [10, 20, 36]
        if iter in update_list:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 1.5

def dchuc_algo():
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    '''
    parameter setting
    '''
    max_iter = opt.max_iter
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    learning_rate_txt = opt.learning_rate_txt
    weight_decay = 5 * 10 ** -4
    num_samples = opt.num_samples
    gamma = opt.gamma
    alpha = opt.alpha
    beta = opt.beta
    yita = opt.yita
    mu = opt.mu
    code_length = opt.bits
    dataname = opt.dataname

    '''
    dataset preprocessing
    '''
    nums, dsets, labels = _dataset(dataname)
    num_database, num_test, num_train = nums
    dset_database, dset_database_txt, dset_test, dset_test_txt, dset_train, dset_train_txt = dsets
    y_dim = dset_database_txt.y_dim
    database_labels, test_labels, train_labels = labels
    n_class = test_labels.size()[1]

    testloader = DataLoader(dset_test, batch_size=1,
                            shuffle=False,
                            num_workers=4)
    testloader_txt = DataLoader(dset_test_txt, batch_size=1,
                                shuffle=False,
                                num_workers=4)
    retrievalloader = DataLoader(dset_database, batch_size=1,
                            shuffle=False,
                            num_workers=4)
    retrievalloader_txt = DataLoader(dset_database_txt, batch_size=1,
                                shuffle=False,
                                num_workers=4)
    '''
    model construction
    '''
    model = cnn_model.CNNNet(opt.arch, code_length)
    model.cuda()
    model_txt = txt_model.TxtModule(y_dim, code_length)
    model_txt.cuda()
    adsh_loss = dl.DCHUCLoss(gamma, code_length, num_database, alpha, mu)
    adsh_loss_txt = dl.DCHUCLoss(gamma, code_length, num_database, alpha, mu)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_txt = optim.SGD(model_txt.parameters(), lr=learning_rate_txt, weight_decay=weight_decay)

    V = np.zeros((num_train, code_length))
    W = np.random.normal(loc=0.0, scale=0.01, size=(code_length, n_class))
    model.train()

    start_time = time.time()
    for iter in range(max_iter):
        '''
        sampling and construct similarity matrix
        '''
        select_index = list(np.random.permutation(range(num_train)))[0: num_samples]
        _sampler = subsetsampler.SubsetSampler(select_index)
        trainloader = DataLoader(dset_train, batch_size=batch_size,
                                 sampler=_sampler,
                                 shuffle=False,
                                 num_workers=4)
        trainloader_txt = DataLoader(dset_train_txt, batch_size=batch_size,
                                     sampler=_sampler,
                                     shuffle=False,
                                     num_workers=4)
        '''
        learning deep neural network: feature learning
        '''
        sample_label = train_labels.index_select(0, torch.from_numpy(np.array(select_index)))
        Sim = calc_sim(sample_label, train_labels)
        S1 = calc_sim(sample_label, sample_label)
        U = np.zeros((num_samples, code_length), dtype=np.float)
        G = np.zeros((num_samples, code_length), dtype=np.float)
        for epoch in range(epochs):
            for zz in range(1):
                for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
                    batch_size_ = train_label.size(0)
                    u_ind = np.linspace(iteration * batch_size, np.min((num_samples, (iteration + 1) * batch_size)) - 1,
                                        batch_size_, dtype=int)
                    train_input = Variable(train_input.cuda())

                    output = model(train_input)
                    S = Sim.index_select(0, torch.from_numpy(u_ind))
                    S_1 = S.index_select(1, torch.from_numpy(u_ind))
                    U[u_ind, :] = output.cpu().data.numpy()

                    model.zero_grad()
                    loss = adsh_loss(output, G[u_ind, :], V, S, S_1, V[batch_ind.cpu().numpy(), :],
                                     Variable(torch.from_numpy(W).type(torch.FloatTensor).cuda()),
                                     Variable(train_label.type(torch.FloatTensor).cuda()))
                    loss.backward()
                    optimizer.step()

            for zz in range(1):
                for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader_txt):
                    batch_size_ = train_label.size(0)
                    u_ind = np.linspace(iteration * batch_size, np.min((num_samples, (iteration + 1) * batch_size)) - 1,
                                         batch_size_, dtype=int)
                    train_input = train_input.unsqueeze(1).unsqueeze(-1).type(torch.FloatTensor)
                    train_input = Variable(train_input.cuda())

                    output = model_txt(train_input)
                    S = Sim.index_select(0, torch.from_numpy(u_ind))
                    S_1 = S.index_select(1, torch.from_numpy(u_ind))
                    G[u_ind, :] = output.cpu().data.numpy()

                    model_txt.zero_grad()
                    loss = adsh_loss_txt(output, U[u_ind, :], V, S, S_1, V[batch_ind.cpu().numpy(), :],
                                         Variable(torch.from_numpy(W).type(torch.FloatTensor).cuda()),
                                         Variable(train_label.type(torch.FloatTensor).cuda()))
                    loss.backward()
                    optimizer_txt.step()
        adjusting_learning_rate(optimizer, iter)
        adjusting_learning_rate(optimizer_txt, iter)
        '''
        learning binary codes: discrete coding
        '''
        barU = np.zeros((num_train, code_length))
        barG = np.zeros((num_train, code_length))
        barU[select_index, :] = U
        barG[select_index, :] = G
        Q = -2 * code_length * Sim.cpu().numpy().transpose().dot(U + G) - gamma * (barU + barG)\
            - 2 * beta * train_labels.numpy().dot(W.transpose())
        for k in range(code_length):
            sel_ind = np.setdiff1d([ii for ii in range(code_length)], k)
            V_ = V[:, sel_ind]
            W_ = W.transpose()[:, sel_ind]
            Wk = W.transpose()[:, k]
            Uk = U[:, k]
            Gk = G[:, k]
            U_ = U[:, sel_ind]
            G_ = G[:, sel_ind]
            V[:, k] = -np.sign(Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk) + 2 * G_.transpose().dot(Gk)
                                                    + beta * 2 * W_.transpose().dot(Wk)))

        I = np.eye(code_length)
        P = np.matrix(U.transpose().dot(U)) * alpha + alpha * np.matrix(G.transpose().dot(G))\
            + beta * np.matrix(V.transpose().dot(V)) + yita * I
        PN = np.linalg.pinv(P)
        BL = (alpha * barU + alpha * barG + beta * V).transpose().dot(train_labels.numpy())
        W = np.asarray(PN.dot(BL))

        lossx = calc_loss(W, sample_label, V, U, G, Sim.cpu().numpy(), S1.cpu().numpy(), code_length, select_index, gamma, yita, alpha, mu)

        print('[Iteration: %3d/%3d][Train Loss: %.4f]' % (iter, max_iter, lossx))
    end_time = time.time()
    print('DCHUC Train time: ', (end_time - start_time))

    '''
    training procedure finishes, evaluation
    '''

    model.eval()
    model_txt.eval()
    start_time = time.time()
    # query
    qB_img = encode(model, testloader, num_test, code_length)
    qB_txt = encode(model_txt, testloader_txt, num_test, code_length, istxt=True)
    # retrieval
    rB_img = encode(model, retrievalloader, num_database, code_length)
    rB_txt = encode(model_txt, retrievalloader_txt, num_database, code_length, istxt=True)

    map_img2txt = calc_hr.calc_map(qB_img, rB_txt, test_labels.numpy(), database_labels.numpy())
    map_txt2img = calc_hr.calc_map(qB_txt, rB_img, test_labels.numpy(), database_labels.numpy())
    print('[Evaluation: mAP_img2txt %.4f, mAP_txt2img: %.4f]' % (map_img2txt, map_txt2img))

    end_time = time.time()
    print('Test time :' , (end_time - start_time))
    
    if opt.FLAG_savecode == 1:
        save_hash_code(qB_txt, qB_img, test_labels.numpy(), rB_txt, rB_img, database_labels.numpy(), dataname, code_length)

    with open('result/' + dataname + '.txt', 'a+') as f:
        f.write('[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n' % (dataname, code_length, map_img2txt, map_txt2img))


if __name__=="__main__":
    if not os.path.exists('result'):
        os.makedirs('result')
    
    dchuc_algo()

