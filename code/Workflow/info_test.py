n_spatial = 100
n_time = 81
x = np.linspace(0, 1, n_spatial)
t = np.linspace(0, 0.8, n_time)
    
X = np.outer(np.sin(np.pi * x), np.exp(-0.1 * t)) + \
    0.3 * np.outer(np.sin(2 * np.pi * x), np.exp(-0.5 * t))

stats = info_data(X, dt=0.01)
