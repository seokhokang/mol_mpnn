import numpy as np
import pickle as pkl
#from sklearn.preprocessing import StandardScaler
import csv, sys
from MPNN import Model


#data
opt='train'

n_max=38
dim_node=7 + 5 + 7 + 6 + 2 + 2
dim_edge=4 + 3 + 2 + 2

atom_list=['H','C','N','O','F','S','Cl']

data_path = './'+opt+'_graph.pkl'
save_dict = './'

print(':: load data')
with open(data_path,'rb') as f:
    [DV, DE, DP, DY, Dsmi] = pkl.load(f)

DV = DV.todense()
DE = DE.todense()
DP = np.expand_dims(DP, 3)

dim_atom = len(atom_list)
dim_y = DY.shape[1]

n_val = 5000
n_trn = len(DV) - n_val

print(DV.shape, DE.shape, DP.shape, DY.shape)

print(':: preprocess data')

def _permutation(set):
    permid = np.random.permutation(len(set[0]))
    for i in range(len(set)):
        set[i] = set[i][permid]

    return set

np.random.seed(134)
[DV, DE, DP, DY, Dsmi] = _permutation([DV, DE, DP, DY, Dsmi])

DV_trn = DV[:n_trn]
DE_trn = DE[:n_trn]
DP_trn = DP[:n_trn]
DY_trn = DY[:n_trn]
    
DV_val = DV[n_trn:n_trn+n_val]
DE_val = DE[n_trn:n_trn+n_val]
DP_val = DP[n_trn:n_trn+n_val]
DY_val = DY[n_trn:n_trn+n_val]

print(DV_trn.shape, DV_val.shape)

model = Model(n_max, dim_node, dim_edge, dim_atom, dim_y, dr=0.2, lr=0.0001)

print(':: train model')
with model.sess:
    save_path=save_dict+'MPNN_model.ckpt'
    model.train(DV_trn, DE_trn, DP_trn, DY_trn, DV_val, DE_val, DP_val, DY_val, save_path)