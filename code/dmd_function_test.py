# Test script for DMD function

xi = np.linspace(-10, 10, 400)
t = np.linspace(0, 4*np.pi, 200)
dt = t[1] - t[0]
r = 2

Xgrid, T = np.meshgrid(xi, t)

f1 = 1/np.cosh(Xgrid + 3) * np.exp(1j*2.3*T)
f2 = (1/np.cosh(Xgrid) * np.tanh(Xgrid)) * (2 * np.exp(1j*2.8*T))

f = f1 + f2
X = f.T   

t_pred = np.linspace(0, 8*np.pi, 400)

Phi, omega, lam, b, Xdmd = DMD(X, r, dt, t_pred)



fig = plt.figure(figsize=(16, 10))

ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax3 = fig.add_subplot(2, 3, 4, projection='3d')
ax4 = fig.add_subplot(2, 3, 5, projection='3d')


ax_extended = fig.add_subplot(1, 3, 3, projection='3d')

def surf(ax, Z, title):
    Y, X_grid = np.mgrid[0:Z.shape[0], 0:Z.shape[1]]
    ax.plot_surface(X_grid, Y, np.real(Z), cmap="gray", edgecolor="none", 
                    rstride=1, cstride=1, shade=True)
    ax.view_init(elev=30, azim=-45)
    ax.set_title(title)
    ax.set_xlabel("space")
    ax.set_ylabel("time")


surf(ax1, f1, "f1 (observed)")
surf(ax2, f2, "f2 (observed)")
surf(ax3, f, "f = f1+f2 (observed)")
surf(ax4, Xdmd.T[:200, :], "Xdmd (observed period)")


Y_ext, X_ext = np.mgrid[0:Xdmd.T.shape[0], 0:Xdmd.T.shape[1]]
surf_plot = ax_extended.plot_surface(X_ext, Y_ext, np.real(Xdmd.T), 
                                      cmap="viridis", edgecolor="none", 
                                      rstride=2, cstride=2, shade=True)


n_observed = X.shape[1]
ax_extended.plot([0, Xdmd.T.shape[0]], [n_observed, n_observed], [0, 0], 
                 'r--', linewidth=3, label='Fin observations')

ax_extended.view_init(elev=30, azim=-45)
ax_extended.set_title("DMD Prediction (Extended to Future)", fontsize=14, fontweight='bold')
ax_extended.set_xlabel("space")
ax_extended.set_ylabel("time")
ax_extended.set_zlabel("amplitude")
ax_extended.legend()

plt.tight_layout()
plt.show()






# Test script for DMD with control with known B matrix

# System parameters from the book example
A = np.array([[1.5, 0], [0, 0.1]])
x0 = np.array([[4], [7]])
K = -1
m = 20
B = np.array([[1], [0]])
dt = 1.0  # time step

# Data collection
DataX = x0
DataU = np.array([[0]])

for j in range(m):
    x_curr = DataX[:, -1:]
    u_curr = K * x_curr[0, 0]
    x_next = A @ x_curr + B * u_curr

    DataX = np.hstack([DataX, x_next])
    DataU = np.hstack([DataU, [[u_curr]]])

DataU = DataU[:, 1:]  # Remove initial zero

# Apply DMDc
A_DMDc, eigs, modes = fu.DMDc_known(DataX, DataU, B, dt, r=None)

print("A_DMDc shape:", A_DMDc.shape)
print("\nA_DMDc matrix:")
print(A_DMDc)
print("\nEigenvalues:")
print(eigs)





# Test script for DMD with control with unknown B matrix

# System parameters (true system - unknown to DMDc)
A_true = np.array([[1.5, 0], [0, 0.1]])
B_true = np.array([[1], [0]])
x0 = np.array([[4], [7]])
K = -1
m = 20
dt = 1.0  # time step

# Data collection with INDEPENDENT random inputs
np.random.seed(42)  # for reproducibility
DataX = x0
DataU = np.empty((1, 0))

for j in range(m):
    x_curr = DataX[:, -1:]
    # Use random independent input instead of state feedback
    u_curr = np.random.randn() * 2.0  # independent random input
    x_next = A_true @ x_curr + B_true * u_curr

    DataX = np.hstack([DataX, x_next])
    DataU = np.hstack([DataU, [[u_curr]]])

print("=" * 60)
print("WITH INDEPENDENT RANDOM INPUTS:")
print("=" * 60)

# Apply DMDc (B unknown)
A_DMDc, B_DMDc, eigs, modes = fu.DMDc_unknown(DataX, DataU, dt, r=None)

print("Identified A_DMDc:")
print(A_DMDc)
print("\nTrue A:")
print(A_true)
print("\nIdentified B_DMDc:")
print(B_DMDc)
print("\nTrue B:")
print(B_true)
print("\nEigenvalues:")
print(eigs)


