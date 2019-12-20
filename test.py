import numpy as np
import pickle as pkl
import csv, sys
from util import _permutation
from MPNN import Model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
scaler.fit(DY)
DY = scaler.transform(DY)

dim_atom = len(atom_list)
dim_y = DY.shape[1]
print(DV.shape, DE.shape, DP.shape, DY.shape)

n_tst = 10000
n_val = 10000
n_trn = len(DV) - n_tst - n_val

print(':: preprocess data')
np.random.seed(134)
[DV, DE, DP, DY, Dsmi] = _permutation([DV, DE, DP, DY, Dsmi])

DV_tst = DV[-n_tst:]
DE_tst = DE[-n_tst:]
DP_tst = DP[-n_tst:]
DY_tst = DY[-n_tst:]

model = Model(n_max, dim_node, dim_edge, dim_atom, dim_y, dr=0, lr=0.0001)
with model.sess:
    model.saver.restore(model.sess, save_path)  
    DY_tst_hat = model.test(DV_tst, DE_tst, DP_tst)
    
np.set_printoptions(precision=5, suppress=True)
maelist = np.array([mean_absolute_error(DY_tst[:,yid:yid+1], DY_tst_hat[:,yid:yid+1]) for yid in range(dim_y)])
mae = np.sum(maelist)
print(':: MAE ', mae, mae/dim_y)
print(':: list ', maelist) 