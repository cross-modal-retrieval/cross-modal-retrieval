import h5py
import numpy as np

def loading_data(dataname):
    if dataname == 'flickr':
        path = '../Data/raw_mir.mat'
    elif dataname == 'nuswide':
        path = '../Data/raw_nus.mat'
    elif dataname == 'coco':
        path = '../Data/raw_coco.mat'  
    else:
        print('Dataname Error!')

    f = h5py.File(path, 'r')
    X, Y, L = {}, {}, {}

    X['train'] = f['I_tr'][:].transpose(3, 0, 1, 2)
    Y['train'] = f['T_tr'][:].T
    L['train'] = f['L_tr'][:].T

    X['query'] = f['I_te'][:].transpose(3, 0, 1, 2)
    Y['query'] = f['T_te'][:].T
    L['query'] = f['L_te'][:].T

    X['retrieval'] = f['I_db'][:].transpose(3, 0, 1, 2)
    Y['retrieval'] = f['T_db'][:].T
    L['retrieval'] = f['L_db'][:].T

    f.close()
    return X, Y, L

