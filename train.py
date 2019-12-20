import numpy as np
import pickle as pkl
import csv, sys
from util import _permutation
from MPNN import Model
from sklearn.preprocessing import StandardScaler


n_max=29
dim_node=25
dim_edge=11
atom_list=['H','C','N','O','F',]

data_path = './QM9_graph.pkl'
save_path = './MPNN_model.ckpt'

print(':: load data')
with open(data_path,'rb') as f:
    [DV, DE, DP, DY, Dsmi] = pkl.load(f)

DV = DV.todense()
DE = DE.todense()
DP = np.expand_dims(DP, 3)

scaler = StandardScaler()
DY = scaler.fit_transform(DY)

dim_atom = len(atom_list)
dim_y = DY.shape[1]
print(DV.shape, DE.shape, DP.shape, DY.shape)

n_tst = 10000
n_val = 10000
n_trn = len(DV) - n_tst - n_val

print(':: preprocess data')
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

model = Model(n_max, dim_node, dim_edge, dim_atom, dim_y, dr=0, lr=0.0001)
with model.sess:
    load_path=None
    model.train(DV_trn, DE_trn, DP_trn, DY_trn, DV_val, DE_val, DP_val, DY_val, load_path, save_path)