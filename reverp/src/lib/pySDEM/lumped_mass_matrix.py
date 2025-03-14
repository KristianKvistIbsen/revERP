import numpy as np

import scipy.sparse as sp

def lumped_mass_matrix(v, f):
    """
    Compute the lumped mass matrix for FEM Laplacian.
    If you use this code in your own work, please cite:
    G.P.T. Choi and C.H. Rycroft, "Density-equalizing maps for simply 
    connected open surfaces." SIAM Journal on Imaging Sciences, 11(2), 
    pp.1134-1178, 2018.
    """
    f1, f2, f3 = f[:,0], f[:,1], f[:,2]
    l1 = np.linalg.norm(v[f2] - v[f3], axis=1)
    l2 = np.linalg.norm(v[f3] - v[f1], axis=1)
    l3 = np.linalg.norm(v[f1] - v[f2], axis=1)
    s = 0.5*(l1 + l2 + l3)
    area = np.sqrt(s*(s-l1)*(s-l2)*(s-l3))

    II = np.hstack([f1, f2, f3])
    JJ = np.hstack([f1, f2, f3])
    V = np.hstack([area, area, area]) / 3.0

    A = sp.coo_matrix((V, (II, JJ)), shape=(len(v), len(v))).tocsr()
    return A