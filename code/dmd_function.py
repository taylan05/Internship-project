import numpy as np

def basicSVD(A):
    # Compute SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Create Sigma with the same shape as A
    Sigma = np.zeros_like(A, dtype=float)
    np.fill_diagonal(Sigma, S)
    
    return U, Sigma, Vt



def SVD(A):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    Sigma = np.diag(S)
    return U, Sigma, Vt





def BasicLeastsqr(X1, X2, r = None):  # Find A to get the minimum of ||X2 - A * X1|| (Frobenius norm), truncation r
    U, S, Vt = SVD(X1)

    if r is None:
        r = U.shape[1]
        
    r = min(r, U.shape[0])
    Ur = U[:, :r]
    Sr = S[:r, :r]
    Vtr = Vt[:r, :].T

    A = X2 @ Vtr @ np.linalg.inv(Sr) @ Ur.T

    return A, Ur, Sr, Vtr
    



def Leastsqr(X1, X2, r=None):

    U, Sigma, Vt = SVD(X1)
    
    S = np.diag(Sigma) 
    
    if r is None:
        r = len(S)
    r = min(r, len(S))
    
    Ur = U[:, :r]
    Sr_diag = S[:r]  
    Sr = np.diag(Sr_diag)
    Vtr = Vt[:r, :]
    
    Sr_inv = np.diag(1.0 / Sr_diag)  
    A = X2 @ Vtr.T @ Sr_inv @ Ur.T
    
    return A, Ur, Sr, Vtr












def DMDcknown(X, Y, B, dt, r = None):
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    Ups = Y[:, :-1]
    Omega = np.vstack((X1, Ups))
    Theta = X2 - B @ Ups

    A, _, Sr, Vtr = Leastsqr(X1, Theta, r)

    return A






def DMDcunknown(X, Y, r = None):
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    Ups = Y[:, :-1]
    Omega = np.vstack((X1, Ups))

    Z, Ur, Sr, Vtr = Leastsqr(Omega, X2, r)
    n = X1.shape[0]  
    q = Ups.shape[0]

    if(r == None):
        A = Z[:, :n]
        B = Z[:, n:n+q]
    else:
        Z_full = Zr @ Vtr.T @ np.linalg.inv(Sr) @ Ur.T
        A = Z_full[:, :n]
        B = Z_full[:, n:n+q]    

    return A, B





def Operinf(X, U, dt):
    l, m = X.shape
    q = U.shape[0]

    Xdot = np.zeros_like(X)
    Xdot[:, :-1] = (X[:, 1:] - X[:, :-1]) / dt
    Xdot[:, -1] = Xdot[:, -2]
    
    Q = np.zeros((l*l, m))
    for i in range(m):
        Q[:, i] = np.kron(X[:, i], X[:, i])

    D = np.vstack((X, Q, U))
    R = Xdot

    O, _, _, _ = Leastsqr(D, R)

    A = O[:, :l]
    H = O[:, l:l + l*l]
    B = O[:, l + l*l:].reshape(l, q)

    return A, H, B







def DMD(X1, X2, r, dt):
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r = min(r, U.shape[1])
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    V_r = Vh.conj().T[:, :r]

    Atilde = U_r.conj().T @ X2 @ V_r @ np.linalg.inv(S_r)

    D, W_r = np.linalg.eig(Atilde)
    Phi = X2 @ V_r @ np.linalg.inv(S_r) @ W_r

    lam = D
    omega = np.log(lam) / dt

    x1 = X1[:, 0]
    b = np.linalg.lstsq(Phi, x1, rcond=None)[0]

    m = X1.shape[1]
    t = np.arange(m) * dt
    time_dynamics = np.zeros((r, m), dtype=complex)
    for k in range(m):
        time_dynamics[:, k] = b * np.exp(omega * t[k])
    Xdmd = Phi @ time_dynamics

    return Phi, omega, lam, b, Xdmd







def DMDperf(X, r, dt, t, xi):
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





def DMD_handmade(X1, X2, r, dt):
    
    U, Sigma, Vt = fu.basicSVD(X1)
    
    n_singular = min(Sigma.shape[0], Sigma.shape[1])
    singular_values = np.array([Sigma[i, i] for i in range(n_singular)])
    
    r = min(r, n_singular)
    U_r = U[:, :r]
    S_r = np.diag(singular_values[:r])
    V_r = Vt[:r, :].conj().T
    
        
    Atilde = U_r.conj().T @ X2 @ V_r @ np.linalg.inv(S_r)
    D, W_r = np.linalg.eig(Atilde)
    
    Phi = X2 @ V_r @ np.linalg.inv(S_r) @ W_r
    
    lam = D
    omega = np.log(lam) / dt
    
    x1 = X1[:, 0]
    b = np.linalg.lstsq(Phi, x1, rcond=None)[0]
    m = X1.shape[1]
    t = np.arange(m) * dt
    
    time_dynamics = np.zeros((r, m), dtype=complex)
    
    for k in range(m):
        time_dynamics[:, k] = b * np.exp(omega * t[k])
    
    Xdmd = Phi @ time_dynamics
    return Phi, omega, lam, b, Xdmd


def DMDc_known(X, U, B, dt, r=None):
    
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
    
    A = (X2 - B @ U) @ V_r @ np.linalg.inv(Sig_r) @ U_r.T
    
    eigenvalues, modes = np.linalg.eig(A)
    
    return A, eigenvalues, modes


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


def DMDc_known_perf(X, U, B, dt, t, r=None):
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
