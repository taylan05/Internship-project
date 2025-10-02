def info_data(X, dt=None):
    """
    Print comprehensive statistics and information about snapshot matrix.
    
    Parameters
    ----------
    X : np.ndarray
        2D snapshot matrix (spatial_dim x time_steps).
    dt : float, optional
        Time step between snapshots.
        
    Returns
    -------
    dict
        Dictionary containing all computed statistics.
    """
    n_spatial, n_time = X.shape
    
    # Compute singular values for rank estimation
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    energy_cumsum = np.cumsum(s**2) / np.sum(s**2)
    
    # Collect statistics
    stats = {
        'n_spatial': n_spatial,
        'n_time': n_time,
        'total_elements': X.size,
        'mean': np.mean(X),
        'std': np.std(X),
        'min': np.min(X),
        'max': np.max(X),
        'full_rank': min(n_spatial, n_time),
        'condition_number': s[0] / s[-1] if s[-1] != 0 else np.inf,
        'memory_mb': X.nbytes / 1e6,
        'dtype': X.dtype,
        'n_nan': np.sum(np.isnan(X)),
        'n_inf': np.sum(np.isinf(X)),
        'singular_values': s
    }
    
    # Print statistics
    
    print(f"\nDimensions:")
    print(f"  Spatial dimension: {n_spatial:,}")
    print(f"  Temporal snapshots: {n_time:,}")
    if dt is not None:
        print(f"  Time step: {dt:.4f}")
    print(f"  Total elements: {X.size:,}")
    
    print(f"\nValue Statistics:")
    print(f"  Mean: {stats['mean']:.4e}")
    print(f"  Std:  {stats['std']:.4e}")
    print(f"  Min:  {stats['min']:.4e}")
    print(f"  Max:  {stats['max']:.4e}")
    
    
    print(f"\nData Quality:")
    print(f"  Memory: {stats['memory_mb']:.2f} MB")
    print(f"  Dtype: {stats['dtype']}")
    print(f"  NaN values: {stats['n_nan']}")
    print(f"  Inf values: {stats['n_inf']}")
    

    return stats
