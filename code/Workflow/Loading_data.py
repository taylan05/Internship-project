import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat


def load_csv(filename, as_dataframe=False):
    """

    If True, return pandas DataFrame. Otherwise return numpy array.
    
    Returns
    
    data : np.ndarray or pd.DataFrame
    """
    df = pd.read_csv(filename)
    return df if as_dataframe else df.values



### If the CSV file is too big, we use chunks


def iter_csv_batches(filename, batch_size=10000):
    """
    Iterate over CSV file in batches.
    
    batch_size : int, default 10000
        Number of rows per batch.
    
    Yields
        chunk : np.ndarray
        Batch of rows as numpy array.
    """
    for chunk in pd.read_csv(filename, chunksize=batch_size):
        yield chunk.values



### Usage ###

# Normal case: just load it
data = load_csv("normal_file.csv")

# Large file: use batches
for batch in iter_csv_batches("huge_file.csv", batch_size=50000):
    process(batch)


#######



def load_excel(filename, as_dataframe=False):
    """
    
    If True, return pandas DataFrame. Otherwise return numpy array.
    
    Returns
        data : np.ndarray or pd.DataFrame
    """
    df = pd.read_excel(filename)
    return df if as_dataframe else df.values


def load_npy(filename, mmap=False):
    """
    Load numpy binary file.
   
   If True, use memory mapping for large files.
    
    Returns
        data : np.ndarray
    """
    return np.load(filename, mmap_mode="r" if mmap else None)


def load_npz(filename, key=None):
    """
    Load numpy compressed archive.
    
    
    Specific array name to load. If None and file contains multiple arrays,
    returns a dictionary of all arrays.
    
    Returns
        data : np.ndarray or dict
    """
    data = np.load(filename)
    
    if key is not None:
        return data[key]
    
    if len(data.files) == 1:
        return data[data.files[0]]
    
    return {k: data[k] for k in data.files}


def load_hdf5(filename, dataset_name):
    """
    Load HDF5 dataset.
       
    Returns
        data : np.ndarray
    """
    with h5py.File(filename, "r") as f:
        return f[dataset_name][:]


def load_mat(filename, variable_name):
    """
    Load MATLAB file.
     
    Returns
        data : np.ndarray
    """
    mat = loadmat(filename)
    return mat[variable_name]


def list_hdf5_datasets(filename):
    """
    List all datasets in an HDF5 file.
    """
    with h5py.File(filename, "r") as f:
        return list(f.keys())


def list_mat_variables(filename):
    """
    List all variables in a MATLAB file.
    """
    mat = loadmat(filename)
    return [k for k in mat.keys() if not k.startswith("__")]
