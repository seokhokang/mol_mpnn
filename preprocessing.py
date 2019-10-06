import numpy as np
import pickle as pkl
import os, sys, sparse
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pds

opt=sys.argv[1]


def to_onehot(val, cat):

    vec = np.zeros(len(cat))
    for i, c in enumerate(cat):
        if val == c: vec[i] = 1

    if np.sum(vec) == 0: print('* exception: missing category', val)
    assert np.sum(vec) == 1

    return vec

def atomFeatures(aid, mol, rings, donor_list, acceptor_list):

    def _rings(aid, rings):

        vec = np.zeros(6)
        for ring in rings:
            if aid in ring and len(ring) <= 8:
                vec[len(ring) - 3] += 1

        return vec

    def _da(aid, donor_list, acceptor_list):

        vec = np.zeros(2)
        if aid in donor_list:
            vec[0] = 1
        elif aid in acceptor_list:
            vec[1] = 1
        
        return vec

    def _chiral(a):
        try:
            vec = to_onehot(a.GetProp('_CIPCode'), ['R','S'])
        except:
            vec = np.zeros(2)
        
        return vec
    
    a = mol.GetAtomWithIdx(aid)
        
    v1 = to_onehot(a.GetSymbol(), atom_list)
    v2 = to_onehot(str(a.GetHybridization()), ['S','SP','SP2','SP3','SP3D','SP3D2'])[1:]
    v3 = [a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(), a.GetTotalNumHs(), a.GetImplicitValence(), a.GetNumRadicalElectrons(), int(a.GetIsAromatic())]
    v4 = _rings(aid, rings)
    v5 = _da(aid, donor_list, acceptor_list)
    v6 = _chiral(a)

    return np.concatenate([v1, v2, v3, v4, v5, v6], axis=0)

def bondFeatures(bid1, bid2, mol, rings):

    bondpath = Chem.GetShortestPath(mol, bid1, bid2)
    bonds = [mol.GetBondBetweenAtoms(bondpath[t], bondpath[t + 1]) for t in range(len(bondpath) - 1)]
    
    samering = 0
    for ring in rings:
        if bid1 in ring and bid2 in ring:
            samering = 1

    if len(bonds)==1:
        v1 = to_onehot(str(bonds[0].GetBondType()), ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
        v2 = to_onehot(str(bonds[0].GetStereo()), ['STEREOZ', 'STEREOE','STEREOANY','STEREONONE'])[:3]
        v3 = [int(bonds[0].GetIsConjugated()), int(bonds[0].IsInRing())]
    else:
        v1 = np.zeros(4)
        v2 = np.zeros(3)
        v3 = np.zeros(2)
    
    v4 = [len(bonds), samering]
        
    return np.concatenate([v1, v2, v3, v4], axis=0)


#data
n_max=38
dim_node=7 + 5 + 7 + 6 + 2 + 2
dim_edge=4 + 3 + 2 + 2

atom_list=['H','C','N','O','F','S','Cl']

molsuppl = np.array(Chem.SDMolSupplier('./'+opt+'_mol.sdf', removeHs=False))
molprops = np.array(pds.read_csv('./'+opt+'_target.csv').as_matrix())

fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

DV = []
DE = []
DP = [] 
DY = []
Dsmi = []
for i, mol in enumerate(molsuppl):

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
    node = np.zeros((n_max, dim_node), dtype=int)
    for j in range(n_atom):
        node[j, :] = atomFeatures(j, mol, rings, donor_list, acceptor_list)
    
    # edge DE
    edge = np.zeros((n_max, n_max, dim_edge), dtype=int)
    for j in range(n_atom - 1):
        for k in range(j + 1, n_atom):
            edge[j, k, :] = bondFeatures(j, k, mol, rings)
            edge[k, j, :] = edge[j, k, :]

    # 3D pos DP
    pos = mol.GetConformer().GetPositions()
    proximity = np.zeros((n_max, n_max))
    proximity[:n_atom, :n_atom] = euclidean_distances(pos)

    # property DY    
    pid = np.where(molprops[:,0]==int(mol.GetProp('name')))[0][0]
    property = molprops[pid, 1:]

    # append
    DV.append(np.array(node))
    DE.append(np.array(edge))
    DP.append(np.array(proximity))
    DY.append(np.array(property))
    Dsmi.append(mol.GetProp('name'))

    if i % 1000 == 0:
        print(i, flush=True)

# np array    
DV = np.asarray(DV, dtype=int)
DE = np.asarray(DE, dtype=int)
DP = np.asarray(DP)
DY = np.asarray(DY)
Dsmi = np.asarray(Dsmi)

# compression
DV = sparse.COO.from_numpy(DV)
DE = sparse.COO.from_numpy(DE)

# save
with open(opt+'_graph.pkl','wb') as fw:
    pkl.dump([DV, DE, DP, DY, Dsmi], fw)