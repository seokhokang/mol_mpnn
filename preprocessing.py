import numpy as np
import pickle as pkl
import os, sys, sparse
from util import atomFeatures, bondFeatures
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pds


# QM9 dataset can be downloaded from
# http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz
molsuppl = Chem.SDMolSupplier('./gdb9.sdf', removeHs=False)
molprops = np.array(pds.read_csv('./gdb9.sdf.csv').as_matrix())

n_max=29
dim_node=25
dim_edge=11
atom_list=['H','C','N','O','F']

rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')

fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

DV = []
DE = []
DP = [] 
DY = []
Dsmi = []
for i, mol in enumerate(molsuppl):

    if mol==None: continue
    try: Chem.SanitizeMol(mol)
    except: continue
    smi = Chem.MolToSmiles(mol)
    if '.' in Chem.MolToSmiles(mol): continue
    
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    Chem.rdmolops.AssignStereochemistry(mol)   
    
    n_atom = mol.GetNumAtoms()
    
    rings = mol.GetRingInfo().AtomRings() 
    
    feats = chem_feature_factory.GetFeaturesForMol(mol)
    donor_list = []
    acceptor_list = []
    for j in range(len(feats)):
        if feats[j].GetFamily() == 'Donor':
            assert len(feats[j].GetAtomIds())==1
            donor_list.append (feats[j].GetAtomIds()[0])
        elif feats[j].GetFamily() == 'Acceptor':
            assert len(feats[j].GetAtomIds())==1
            acceptor_list.append (feats[j].GetAtomIds()[0])
    
    # node DV
    node = np.zeros((n_max, dim_node), dtype=np.int8)
    for j in range(n_atom):
        node[j, :] = atomFeatures(j, mol, rings, donor_list, acceptor_list)
    
    # edge DE
    edge = np.zeros((n_max, n_max, dim_edge), dtype=np.int8)
    for j in range(n_atom - 1):
        for k in range(j + 1, n_atom):
            edge[j, k, :] = bondFeatures(j, k, mol, rings)
            edge[k, j, :] = edge[j, k, :]

    # 3D pos DP
    pos = mol.GetConformer().GetPositions()
    proximity = np.zeros((n_max, n_max))
    proximity[:n_atom, :n_atom] = euclidean_distances(pos)

    # property DY    
    pid = np.where(molprops[:,0]=='gdb_'+str(i+1))[0][0]
    property = molprops[pid, 4:16]

    # append
    DV.append(np.array(node))
    DE.append(np.array(edge))
    DP.append(np.array(proximity))
    DY.append(np.array(property))
    Dsmi.append(smi)

    if i % 1000 == 0:
        print(i+1, Chem.MolToSmiles(Chem.RemoveHs(mol)), property, flush=True)

# np array    
DV = np.asarray(DV, dtype=np.int8)
DE = np.asarray(DE, dtype=np.int8)
DP = np.asarray(DP)
DY = np.asarray(DY)
Dsmi = np.asarray(Dsmi)

# compression
DV = sparse.COO.from_numpy(DV)
DE = sparse.COO.from_numpy(DE)

print(DV.shape, DE.shape, DP.shape, DY.shape)

# save
with open('QM9_graph.pkl','wb') as fw:
    pkl.dump([DV, DE, DP, DY, Dsmi], fw)