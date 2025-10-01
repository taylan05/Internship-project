import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import lstsq
from findiff import Diff
from typing import Tuple
import matplotlib.pyplot as plt

def operator_inference(X, U, dt, r=None, regularization=None, rcond=1e-15):
    
    n, k = X.shape
    

    U_svd, S, Vh = np.linalg.svd(X, full_matrices=False)
    if r is None:
        r = np.sum(S > 1e-10 * S[0])
    else:
        r = min(r, len(S))
    
    V_r = U_svd[:, :r]
    X_r = V_r.T @ X
    

    accuracy = 2
    d_dx = Diff(1, dt, acc=accuracy)
    dX_r = d_dx(X_r)
    
    
    # Data matrix
    X2_r = X_r * X_r
    D = np.concatenate([X_r.T, X2_r.T, U.T], axis=1)
    RHS = dX_r.T


    # Solve
    if regularization is not None :
        DReg = np.vstack([D, np.sqrt(regularization) * np.eye(D.shape[1])])
        RHSReg = np.vstack([RHS, np.zeros((D.shape[1], r))])
        O = lstsq(DReg, RHSReg, cond=rcond)[0].T
    else:
        O = lstsq(D, RHS, cond=rcond)[0].T
    
    A_r = O[:, :r]
    H_r = O[:, r:2*r]
    B_r = O[:, 2*r:].flatten() if O[:, 2*r:].shape[1] == 1 else O[:, 2*r:]
    
    # Diagnostics
    dX_r_pred = A_r @ X_r + H_r @ X2_r + (B_r.reshape(-1, 1) if B_r.ndim == 1 else B_r) @ U
    fit_error = np.linalg.norm(dX_r - dX_r_pred, 'fro') / np.linalg.norm(dX_r, 'fro')
    
    print(f"r={r}, fit_error={fit_error:.4f}, cond(H)={np.linalg.cond(H_r):.2e}")
    
    return A_r, H_r, B_r, V_r
