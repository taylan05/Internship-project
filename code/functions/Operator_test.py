import numpy as np
import matplotlib.pyplot as plt
from findiff import Diff
from scipy.integrate import solve_ivp
from ipywidgets import interact, IntSlider


def burgers_discretization(n, mu):
    h = 1/(n+1)
    k1 = mu/(h**2)
    k2 = 1/(2*h)
    
    A = np.zeros((n, n))
    H = np.zeros((n, n))
    
    for i in range(1, n-1):
        A[i, i-1] = k1
        A[i, i] = -2*k1
        A[i, i+1] = k1
        
        H[i, i-1] = k2
        H[i, i+1] = -k2
    
    # Boundary conditions
    A[-1, -1] = -k1  # Neumann at right
    
    B = np.zeros(n)
    B[0] = k1
    
    return A, H, B

def burgers_model(A, H, B, inpfunc):
    def model(t, x):
        return A @ x + H @ (x * x) + B * inpfunc(t)
    return model






n = 100  # spatial discretization
mu = 0.01  # diffusion coefficient
T = 1.0  # end time
dt = 0.01  # time step
Tsnap = 0.8  # training data end time

k = int(T/dt + 1)
tspan = np.linspace(0, T, k)
snap = int(Tsnap/dt + 1)

inifunc = lambda x: np.sin(np.pi * x)
inpfunc = lambda t: 0.01 * np.sin(0.01 * np.pi * t)

A, H, B = burgers_discretization(n, mu)
burg = burgers_model(A, H, B, inpfunc)

space_grid = np.linspace(1/(n+1), n/(n+1), n)
inival = inifunc(space_grid)

sol = solve_ivp(burg, (0, T), inival, method='BDF', t_eval=tspan)
snapshots = sol.y
training_snapshots = snapshots[:, :snap]
utrain = inpfunc(tspan[:snap]).reshape(1, -1)




r = 7
A_r, H_r, B_r, V_r, O = operator_inference(
    training_snapshots, utrain, dt, r=r, regularization=1e-6, rcond=1e-6
)

inivalr = V_r.T @ inival





def reduced_burg_model(t, xr):
    return A_r @ xr + H_r @ (xr * xr) + B_r * inpfunc(t)

reduced_sol = solve_ivp(reduced_burg_model, (0, T), inivalr, method='BDF', t_eval=tspan)
reduced_snapshots = V_r @ reduced_sol.y




error = np.linalg.norm(snapshots - reduced_snapshots, axis=0) / np.linalg.norm(snapshots, axis=0)

plt.figure(figsize=(10,5))
plt.semilogy(tspan, error, 'k-', linewidth=2)
plt.axvline(Tsnap, color='g', linestyle=':', linewidth=2, label='Training End')
plt.xlabel('Time')
plt.ylabel('Relative Error')
plt.title('Relative Error: ||x_full - x_reduced|| / ||x_full||')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




def opinf_burgers_reg(r, Dmat, RHS, lam):
    DmatReg = np.vstack([Dmat, lam*np.eye(Dmat.shape[1])])
    RHSReg = np.vstack([RHS, np.zeros((2*r+1,r))])
    Opmat = np.linalg.lstsq(DmatReg, RHSReg, rcond=-1)[0].T
    Ar = Opmat[:, :r]
    Hr = Opmat[:, r:2*r]
    Br = Opmat[:, 2*r:].reshape(-1)
    return Ar, Hr, Br, Opmat






accuracy = 2
d_dx = Diff(1, dt, acc=accuracy)
X_r = V_r.T @ training_snapshots
dX_r = d_dx(X_r)
X2_r = X_r * X_r
Dmat = np.concatenate([X_r.T, X2_r.T, utrain.T], axis=1)
RHS = dX_r.T

numb = 150
lreg = np.logspace(-12, 2, num=numb)
residuum = np.zeros(numb)
normsol = np.zeros(numb)

for k, lam in enumerate(lreg):
    DReg = np.vstack([Dmat, np.sqrt(lam)*np.eye(Dmat.shape[1])])
    RHSReg = np.vstack([RHS, np.zeros((Dmat.shape[1], r))])
    O = np.linalg.lstsq(DReg, RHSReg, rcond=1e-12)[0].T
    residuum[k] = np.linalg.norm(Dmat @ O.T - RHS, 'fro')
    normsol[k] = np.linalg.norm(O, 'fro')


plt.figure(figsize=(5,4))
plt.loglog(residuum, normsol, 'b-')
plt.xlabel(r'$\| D O^T - R \|_F$')
plt.ylabel(r'$\| O \|_F$')
plt.title("L-curve")
plt.grid(True, which='both')
plt.show()



ind = 30
reg_param = lreg[ind]
Ar_reg, Hr_reg, Br_reg, _ = opinf_burgers_reg(r, Dmat, RHS, reg_param)


def reduced_burg_model_reg(t, xr):
    return Ar_reg @ xr + Hr_reg @ (xr * xr) + Br_reg * inpfunc(t)

reduced_snapshots_reg = solve_ivp(
    reduced_burg_model_reg, (0, T), inivalr, method='BDF', t_eval=tspan
).y

opinf_reg_snapshots = V_r @ reduced_snapshots_reg



relerr_field = np.abs(snapshots - opinf_reg_snapshots) / np.linalg.norm(snapshots, axis=0)


plt.figure(figsize=(8,5))
plt.contourf(tspan, space_grid, relerr_field, levels=50, cmap='viridis')
plt.colorbar(label="Relative error")
plt.xlabel("Time")
plt.ylabel("Space")
plt.title(r"Relative error distribution $\varepsilon(x,t)$")
plt.show()




times_to_plot = [0.1, 0.5, 0.9]
fig, ax = plt.subplots(figsize=(8,4))
for tt in times_to_plot:
    idx_t = np.argmin(np.abs(tspan - tt))
    ax.plot(space_grid, snapshots[:, idx_t], 'b-', linewidth=2, label=f"Full {tt:.1f}")
    ax.plot(space_grid, reduced_snapshots[:, idx_t], 'r--', linewidth=2, label=f"OpInf {tt:.1f}")
    ax.plot(space_grid, opinf_reg_snapshots[:, idx_t], 'm-.', linewidth=2, label=f"Reg OpInf {tt:.1f}")
ax.set_xlabel("Space")
ax.set_ylabel("State")
ax.set_title("Trajectory comparison")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()




error = np.linalg.norm(snapshots - reduced_snapshots, axis=0) / np.linalg.norm(snapshots, axis=0)
opinf_error = np.linalg.norm(snapshots - opinf_reg_snapshots, axis=0) / np.linalg.norm(snapshots, axis=0)

plt.figure(figsize=(8,4))
plt.semilogy(tspan, error, 'r--', linewidth=2, label='OpInf')
plt.semilogy(tspan, opinf_error, 'm-.', linewidth=2, label='Reg OpInf')
plt.axvline(Tsnap, color='g', linestyle=':', linewidth=2, label='Training End')
plt.grid(True, alpha=0.3)
plt.xlabel('Time')
plt.ylabel(r'$\varepsilon(t)$')
plt.title("Relative Error over Time")
plt.legend()
plt.show()

print(f"\nOpInf - Final relative error: {error[-1]:.4e}")
print(f"OpInf - Max relative error: {np.max(error):.4e}")
print(f"OpInf - Mean relative error: {np.mean(error):.4e}")

print(f"\nReg OpInf - Final relative error: {opinf_error[-1]:.4e}")
print(f"Reg OpInf - Max relative error: {np.max(opinf_error):.4e}")
print(f"Reg OpInf - Mean relative error: {np.mean(opinf_error):.4e}")
