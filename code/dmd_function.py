import numpy as np



def DMD(X, r, dt, t):
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = Vh.conj().T[:, :r]

    Atilde = Ur.conj().T @ X2 @ Vr @ np.linalg.inv(Sr)
    eigvals, W = np.linalg.eig(Atilde)
    Phi = X2 @ Vr @ np.linalg.inv(Sr) @ W

    lam = eigvals
    omega = np.log(lam) / dt

    x1 = X[:, 0]
    b = np.linalg.lstsq(Phi, x1, rcond=None)[0]

    time_dynamics = np.zeros((r, len(t)), dtype=complex)
    for i in range(len(t)):
        time_dynamics[:, i] = b * np.exp(omega * t[i])

    X_dmd = Phi @ time_dynamics

    return Phi, omega, lam, b, X_dmd






def DMDc_unknown(X, U, dt, r=None):
    
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    Omega = np.vstack([X1, U])
    
    U_svd, Sig, Vt = np.linalg.svd(Omega, full_matrices=False)
    V = Vt.T
    
    if r is None:
        thresh = 1e-10
        r = np.sum(Sig > thresh)
    
    U_r = U_svd[:, :r]
    Sig_r = np.diag(Sig[:r])
    V_r = V[:, :r]
    
    G = X2 @ V_r @ np.linalg.inv(Sig_r) @ U_r.T
    n = X1.shape[0]
    
    A_DMDc = G[:, :n]
    B_DMDc = G[:, n:]
    
    eigenvalues, modes = np.linalg.eig(A_DMDc)
    
    return A_DMDc, B_DMDc, eigenvalues, modes


def DMDc_known(X, U, B, dt, t, r=None):
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    U_svd, Sig, Vt = np.linalg.svd(X1, full_matrices=False)
    V = Vt.T
    
    if r is None:
        thresh = 1e-10
        r = np.sum(Sig > thresh)
    
    U_r = U_svd[:, :r]
    Sig_r = np.diag(Sig[:r])
    V_r = V[:, :r]
    
    A_DMDc = (X2 - B @ U) @ V_r @ np.linalg.inv(Sig_r) @ U_r.T
    
    lam, W = np.linalg.eig(A_DMDc)
    Phi = (X2 - B @ U) @ V_r @ np.linalg.inv(Sig_r) @ W
    omega = np.log(lam) / dt
    
    x1 = X[:, 0]
    b = np.linalg.lstsq(Phi, x1, rcond=None)[0]
    
    time_dynamics = np.zeros((r, len(t)), dtype=complex)
    
    for i in range(len(t)):
        time_dynamics[:, i] = b * np.exp(omega * t[i])
    
    X_dmd = Phi @ time_dynamics
    
    X_sim = np.zeros((X.shape[0], len(t)), dtype=float)
    X_sim[:, 0:1] = X[:, 0:1]
    
    if U.shape[1] >= len(t) - 1:
        for i in range(len(t) - 1):
            X_sim[:, i+1:i+2] = A_DMDc @ X_sim[:, i:i+1] + B @ U[:, i:i+1]
    else:
        for i in range(min(U.shape[1], len(t) - 1)):
            X_sim[:, i+1:i+2] = A_DMDc @ X_sim[:, i:i+1] + B @ U[:, i:i+1]
        for i in range(U.shape[1], len(t) - 1):
            X_sim[:, i+1:i+2] = A_DMDc @ X_sim[:, i:i+1]
    
    return Phi, omega, lam, b, X_dmd, A_DMDc, X_sim
