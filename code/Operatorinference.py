# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python (Internship env)
#     language: python
#     name: internship_env
# ---

# +
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.integrate
import scipy.linalg as la
import matplotlib.pyplot as plt

import opinf

opinf.utils.mpl_config()

import matplotlib as mpl

mpl.rcParams["text.usetex"] = False


# Construct the spatial domain.
L = 1
n = 2**10 - 1
x_all = np.linspace(0, L, n + 2)
x = x_all[1:-1]
dx = x[1] - x[0]

# Construct the temporal domain.
t0, tf = 0, 1
k = 401
t = np.linspace(t0, tf, k)
dt = t[1] - t[0]

# Construct the full-order state matrix A.
diags = np.array([1, -2, 1]) / (dx**2)
A = scipy.sparse.diags(diags, [-1, 0, 1], (n, n))

# Construct the initial condition for the training data.
q0 = x * (1 - x)


def full_order_solve(initial_condition, time_domain):
    """Solve the full-order model with SciPy."""
    return scipy.integrate.solve_ivp(
        fun=lambda t, q: A @ q,
        t_span=[time_domain[0], time_domain[-1]],
        y0=initial_condition,
        t_eval=time_domain,
        method="BDF",
    ).y


# Solve the full-order model to obtain training snapshots.
with opinf.utils.TimedBlock("Full-order solve"):
    Q = full_order_solve(q0, t)

print(f"\nSpatial domain size:\t{x.shape=}")
print(f"Spatial step size:\t{dx=:.10f}")
print(f"\nTime domain size:\t{t.shape=}")
print(f"Temporal step size:\t{dt=:f}")
print(f"\nFull-order matrix A:\t{A.shape=}")
print(f"\nInitial condition:\t{q0.shape=}")
print(f"\nTraining snapshots:\t{Q.shape=}")


def plot_heat_data(Z, title, ax=None):
    """Visualize temperature data in space and time."""
    if ax is None:
        _, ax = plt.subplots(1, 1)

    # Plot a few snapshots over the spatial domain.
    sample_columns = [0, 2, 5, 10, 20, 40, 80, 160, 320]
    color = iter(plt.cm.viridis_r(np.linspace(0.05, 1, len(sample_columns))))

    leftBC, rightBC = [0], [0]
    for j in sample_columns:
        q_all = np.concatenate([leftBC, Z[:, j], rightBC])
        ax.plot(x_all, q_all, color=next(color), label=rf"$q(x,t_{{{j}}})$")

    ax.set_xlim(x_all[0], x_all[-1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$q(x,t)$")
    ax.legend(loc=(1.05, 0.05))
    ax.set_title(title)


plot_heat_data(Q, "Snapshot data")


# Define the reduced-order model.
rom = opinf.ROM(
    basis=opinf.basis.PODBasis(cumulative_energy=0.9999),
    ddt_estimator=opinf.ddt.UniformFiniteDifferencer(t, "ord6"),
    model=opinf.models.ContinuousModel(
        operators="A",
        solver=opinf.lstsq.L2Solver(regularizer=1e-8),
    ),
)

# Calibrate the reduced-order model to data.
rom.fit(Q)

# Solve the reduced-order model.
with opinf.utils.TimedBlock("Reduced-order solve"):
    Q_ROM = rom.predict(q0, t, method="BDF", max_step=dt)

# Compute the relative error of the ROM solution.
opinf.post.frobenius_error(Q, Q_ROM)[1]


# Initialize a basis.
basis = opinf.basis.PODBasis(cumulative_energy=0.9999)

# Fit the basis (compute Vr) using the snapshot data.
basis.fit(Q)
print(basis)

# Visualize the basis vectors.
basis.plot1D(x)
plt.show()


# Compress the state snapshots to the reduced space defined by the basis.
Q_ = basis.compress(Q)

print(f"{Q.shape=}, {Q_.shape=}")


basis.projection_error(Q)


# Compute exact time derivatives using the FOM and compress them.
Qdot_exact = basis.compress(A @ Q)

# Estimate time derivatives using 6th-order finite differences.
ddt_estimator = opinf.ddt.UniformFiniteDifferencer(t, "ord6")
Qdot_ = ddt_estimator.estimate(Q_)[1]

print(f"{Qdot_exact.shape=}\t{Qdot_.shape=}")


# Check that the estimate is close to the true time derivatives.
la.norm(Qdot_exact - Qdot_, ord=np.inf) / la.norm(Qdot_exact, ord=np.inf)


model = opinf.models.ContinuousModel("A")
print(model)


model.fit(states=Q_, ddts=Qdot_exact)
print(model)


# Construct the intrusive ROM linear operator.
Vr = basis.entries
A_intrusive = Vr.T @ A @ Vr

# Compare the OpInf ROM linear operator to the intrusive one.
A_opinf = model.operators[0].entries
np.allclose(A_intrusive, A_opinf)


# Construct the OpInf ROM with estimated time derivatives.
model.fit(states=Q_, ddts=Qdot_)
A_opinf = model.operators[0].entries

np.allclose(A_intrusive, A_opinf)

# Check the difference between intrusive projection and OpInf.
la.norm(A_intrusive - A_opinf) / la.norm(A_intrusive)

# Define a solver for the Tikhonov-regularized least-squares problem.
model = opinf.models.ContinuousModel(
    "A",
    solver=opinf.lstsq.L2Solver(regularizer=1e-2),
)

# Construct the OpInf ROM through regularized least squares.
model.fit(states=Q_, ddts=Qdot_exact)
A_opinf = model.operators[0].entries

# Compare to the intrusive model.
np.allclose(A_intrusive, A_opinf)


# Check the difference between intrusive projection and OpInf.
la.norm(A_intrusive - A_opinf) / la.norm(A_intrusive)


q0_ = basis.compress(q0)  # Compress the initial conditions.

model = opinf.models.ContinuousModel(
    "A",
    solver=opinf.lstsq.L2Solver(regularizer=1e-8),
).fit(Q_, Qdot_)

Q_ROM_ = model.predict(q0_, t, method="BDF")

print(f"{Q_ROM_.shape=}")


Q_ROM = basis.decompress(Q_ROM_)

print(f"{Q_ROM.shape=}")


rom = opinf.ROM(
    basis=opinf.basis.PODBasis(cumulative_energy=0.9999),
    ddt_estimator=opinf.ddt.UniformFiniteDifferencer(t, "ord6"),
    model=opinf.models.ContinuousModel(
        operators="A",
        solver=opinf.lstsq.L2Solver(regularizer=1e-8),
    ),
)

print(rom)


rom.fit(Q)

print(rom)

Q_ROM_2 = rom.predict(q0, t, method="BDF")

np.all(Q_ROM_2 == Q_ROM)


fig, [ax1, ax2] = plt.subplots(1, 2)
plot_heat_data(Q, "Snapshot data", ax1)
plot_heat_data(Q_ROM, "ROM state output", ax2)
ax1.legend([])
plt.show()


abs_l2err, norm_l2err = opinf.post.lp_error(Q, Q_ROM, normalize=True)
fig, ax = plt.subplots(1, 1)
ax.semilogy(t, abs_l2err, "-", label=r"Absolute $\ell^2$ error")
ax.semilogy(t, norm_l2err, "--", label=r"Normalized absolute $\ell^2$ error")
ax.set_xlabel(r"$t$")
ax.set_ylabel("error")
ax.legend(loc="lower left")
plt.show()


abs_froerr, rel_froerr = opinf.post.frobenius_error(Q, Q_ROM)
print(f"Relative Frobenius-norm error: {rel_froerr:%}")

with opinf.utils.TimedBlock("Full-order solve"):
    full_order_solve(q0, t)


with opinf.utils.TimedBlock("Reduced-order solve"):
    rom.predict(q0, t, method="BDF")

n_trials = 10

with opinf.utils.TimedBlock(f"{n_trials} FOM solves") as fomtime:
    for _ in range(n_trials):
        full_order_solve(q0, t)

with opinf.utils.TimedBlock(f"{n_trials} ROM solves") as romtime:
    for _ in range(n_trials):
        rom.predict(q0, t, method="BDF")

print(f"Average FOM time: {fomtime.elapsed / n_trials :.6f} s")
print(f"Average ROM time: {romtime.elapsed / n_trials :.6f} s")
print(f"ROM speedup: {fomtime.elapsed / romtime.elapsed :.4f} times!")

# -


