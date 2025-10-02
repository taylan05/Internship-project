import numpy as np
import matplotlib.pyplot as plt

def visualize_snapshots(X, dt=None, title="Snapshot Matrix Visualization"):
    n_spatial, n_time = X.shape
    time_array = np.arange(n_time) * (dt if dt is not None else 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title}\nShape: {X.shape} (spatial × temporal)",
                 fontsize=14, fontweight='bold')
    
    # Snapshot heatmap (space-time evolution)
    ax = axes[0]
    im = ax.imshow(X, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_title("Space-Time Evolution")
    ax.set_xlabel("Time Index" if dt is None else "Time")
    ax.set_ylabel("Spatial Index")
    plt.colorbar(im, ax=ax, label='State Value')
    
    # Singular value decay
    ax = axes[1]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    energy = np.cumsum(s**2) / np.sum(s**2)
    
    ax.semilogy(s, 'o-', linewidth=2, markersize=4, label='Singular values')
    ax.set_title("Singular Value Decay")
    ax.set_xlabel("Mode Index")
    ax.set_ylabel("Singular Value")
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    # Secondary axis for cumulative energy
    ax2 = ax.twinx()
    ax2.plot(energy, 'r--', linewidth=1.5, alpha=0.7, label='Cumulative energy')
    ax2.set_ylabel("Cumulative Energy", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim([0, 1])
    ax2.legend(loc='center right')
    
    # Spatial profiles
    ax = axes[2]
    n_snapshots_viz = min(5, n_time)
    snapshot_indices = np.linspace(0, n_time-1, n_snapshots_viz, dtype=int)
    for idx in snapshot_indices:
        label = f"t={time_array[idx]:.2f}" if dt else f"t={idx}"
        ax.plot(X[:, idx], alpha=0.7, linewidth=2, label=label)
    ax.set_title(f"Spatial Profiles\n({n_snapshots_viz} time instances)")
    ax.set_xlabel("Spatial Index")
    ax.set_ylabel("State Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig
