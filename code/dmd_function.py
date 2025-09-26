import numpy as np

def SVD(A):
    # Compute SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Create Sigma with the same shape as A
    Sigma = np.zeros_like(A, dtype=float)
    np.fill_diagonal(Sigma, S)
    
    return U, Sigma, Vt






def Leastsqr(X1, X2, r = None):  # Find A to get the minimum of ||X2 - A * X1|| (Frobenius norm), truncation r
    U, S, Vt = SVD(X1)

    if r is None:
        r = U.shape[1]
        
    r = min(r, U.shape[0])
    Ur = U[:, :r]
    Sr = S[:r, :r]
    Vtr = Vt[:r, :].T

    A = X2 @ Vtr @ np.linalg.inv(Sr) @ Ur.T

    return A, Ur, Sr, Vtr
    



def DMD(X, r = None):
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    A, _, _, _ = Leastsqr(X1, X2, r)

    return A




def DMDcknown(X, Y, B, dt, r = None):
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    Ups = Y[:, :-1]
    Omega = np.vstack((X1, Ups))
    Theta = X2 - B @ Ups

    A, _, _, _ = Leastsqr(X1, Theta, r)

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


















def DMD_matlab(X, dt, r = None):
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    A, _, Sr, Vtr = Leastsqr(X1, X2, r)

    eigvals, Wr = np.linalg.eig(A)
    D = np.diag(eigvals)
    Phi = X2 @ Vtr @ np.linalg.inv(Sr) @ Wr

    omega = np.log(eigvals)/dt
    x1 = X[:, 0]
    b = np.linalg.pinv(Phi) @ x1

    mm1 = X1.shape[1]
    time_dynamics = np.zeros((r, mm1), dtype = complex)
    t = np.arange(mm1) * dt

    for i in range(mm1) :
        time_dynamics[:, i] = b * np.exp(omega * t[i])

    Xdmd = Phi @ time_dynamics

    return Atilde, Phi, omega, eigvals, b, Xdmd







def DMDperf_matlab(X, dt, t, r = None):
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    A, _, Sr, Vtr = Leastsqr(X1, X2, r)

    eigvals, Wr = np.linalg.eig(A)
    D = np.diag(eigvals)
    Phi = X2 @ Vtr @ np.linalg.inv(Sr) @ Wr

    tol = 1e-10
    mask = np.abs(eigvals) > tol
    eigvals = eigvals[mask]
    Phi = Phi[:, mask]
    omega = np.log(eigvals) / dt

    x1 = X[:, 0]
    b = np.linalg.pinv(Phi) @ x1
    time = t.shape[0]

    time_dynamics = np.zeros((eigvals.shape[0], time), dtype = complex)
    for i in range(time) :
        time_dynamics[:, i] = b * np.exp(omega * t[i])

    Xdmd = Phi @ time_dynamics

    return Phi, omega, eigvals, b, Xdmd






def DMDc__unknown_matlab(X, Y, dt, thresh):
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    Ups = Y[:, :-1]
    Omega = np.vstack((X1, Ups))

    U, S, Vt = SVD(Omega)
    rtil = np.sum(np.diag(S) > thresh)

    Util   = U[:, :rtil]
    Stil = S[:rtil, :rtil]
    Vtil   = Vt[:rtil, :].T

    Up, Sp, Vtp = SVD(X2)
    r = np.sum(np.diag(Sp) > thresh)

    Uhat   = Up[:, :r]
    Shat = Sp[:r, :r]
    Vhat   = Vtp[:r, :].T

    n = X1.shape[0]
    q = Ups.shape[0]

    U1 = Util[:n, :]
    U2 = Util[n:n+q, :]

    approxA = Uhat.T @ X2 @ Vtil @ np.linalg.inv(Stil) @ U1.T @ Uhat
    approxB = Uhat.T @ X2 @ Vtil @ np.linalg.inv(Stil) @ U2.T

    eigvals, W = np.linalg.eig(approxA)

    Phi = X2 @ Vtil @ np.linalg.inv(Stil) @ U1.T @ Uhat @ W

    omega = np.log(eigvals) / dt

    x1 = X[:, 0]
    b = np.linalg.pinv(Phi) @ x1

    m = X.shape[1]
    time_dynamics = np.zeros((r, m), dtype=complex)
    for i in range(m):
        time_dynamics[:, i] = b * np.exp(omega * i * dt)

    Xdmd = Phi @ time_dynamics
    
    return Phi, eigvals, approxA, approxB, Xdmd

    



def DMDc_unknown_perf_matlab(X, Y, dt, t, thresh):
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    Ups = Y[:, :-1]
    Omega = np.vstack((X1, Ups))

    U, S, Vt = SVD(Omega)
    rtil = np.sum(np.diag(S) > thresh)

    Util   = U[:, :rtil]
    Stil = S[:rtil, :rtil]
    Vtil   = Vt[:rtil, :].T

    Up, Sp, Vtp = SVD(X2)
    r = np.sum(np.diag(Sp) > thresh)

    Uhat   = Up[:, :r]
    Shat = Sp[:r, :r]
    Vhat   = Vtp[:r, :].T

    n = X1.shape[0]
    q = Ups.shape[0]

    U1 = Util[:n, :]
    U2 = Util[n:n+q, :]

    approxA = Uhat.T @ X2 @ Vtil @ np.linalg.inv(Stil) @ U1.T @ Uhat
    approxB = Uhat.T @ X2 @ Vtil @ np.linalg.inv(Stil) @ U2.T

    eigvals, W = np.linalg.eig(approxA)

    Phi = X2 @ Vtil @ np.linalg.inv(Stil) @ U1.T @ Uhat @ W

    omega = np.log(eigvals) / dt

    x1 = X[:, 0]
    b = np.linalg.pinv(Phi) @ x1

    time = t.shape[0]

    time_dynamics = np.zeros((eigvals.shape[0], time), dtype = complex)
    for i in range(time) :
        time_dynamics[:, i] = b * np.exp(omega * t[i])

    Xdmd = Phi @ time_dynamics

    return Phi, eigvals, approxA, approxB, Xdmd





    