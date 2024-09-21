import h5py
import numpy as np

'''
def loading_data(path):
	print('******************************************************')
	print('dataset:{0}'.format(path))
	print('******************************************************')

	file = h5py.File(path)
	images = file['IAll'][:].transpose(0,3,2,1)
	labels = file['LAll'][:].transpose(1,0)
	tags = file['YAll'][:].transpose(1,0)
	file.close()

	return images, tags, labels
'''

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

def split_data(images, tags, labels, QUERY_SIZE, TRAINING_SIZE, DATABASE_SIZE):

	X = {}
	index_all = np.random.permutation(QUERY_SIZE+DATABASE_SIZE)
	ind_Q = index_all[0:QUERY_SIZE]
	ind_T = index_all[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE]
	ind_R = index_all[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE]

	X['query'] = images[ind_Q, :, :, :]
	X['train'] = images[ind_T, :, :, :]
	X['retrieval'] = images[ind_R, :, :, :]

	Y = {}
	Y['query'] = tags[ind_Q, :]
	Y['train'] = tags[ind_T, :]
	Y['retrieval'] = tags[ind_R, :]

	L = {}
	L['query'] = labels[ind_Q, :]
	L['train'] = labels[ind_T, :]
	L['retrieval'] = labels[ind_R, :]
	return X, Y, L
