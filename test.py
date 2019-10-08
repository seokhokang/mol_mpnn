import numpy as np
import pickle as pkl
#from sklearn.preprocessing import StandardScaler
import csv, sys
from MPNN import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error


#data
opt='test'

n_max=38
dim_node=7 + 5 + 7 + 6 + 2 + 2
dim_edge=4 + 3 + 2 + 2

atom_list=['H','C','N','O','F','S','Cl']

data_path = './'+opt+'_graph.pkl'
save_path = './MPNN_model.ckpt'

print(':: load data')
with open(data_path,'rb') as f:
    [DV, DE, DP, DY, Dsmi] = pkl.load(f)

DV = DV.todense()
DE = DE.todense()
DP = np.expand_dims(DP, 3)

dim_atom = len(atom_list)
dim_y = DY.shape[1]

print(DV.shape, DE.shape, DP.shape, DY.shape)

model = Model(n_max, dim_node, dim_edge, dim_atom, dim_y, dr=0.2, lr=0.0001, batch_size=1)
np.set_printoptions(precision=3, suppress=True)

with model.sess:
    model.saver.restore(model.sess, save_path)  
    DY_hat = model.test(DV, DE, DP)
    
    maelist = [mean_absolute_error(DY[:,yid:yid+1], DY_hat[:,yid:yid+1]) for yid in range(dim_y)]
    mae = np.sum(maelist)
    print(':: MAE ', mae, mae/12)
    print(':: list ', maelist) 