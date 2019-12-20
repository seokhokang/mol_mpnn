import numpy as np
from rdkit import Chem

n_max=29
dim_node=25
dim_edge=11
atom_list=['H','C','N','O','F',]

def _permutation(set):
    permid = np.random.permutation(len(set[0]))
    for i in range(len(set)):
        set[i] = set[i][permid]

    return set

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
    v2 = to_onehot(str(a.GetHybridization()), ['S','SP','SP2','SP3'])[1:]
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