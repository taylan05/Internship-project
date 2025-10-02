# For CSV/TXT files (small to medium dataset)
# The use of chunk is needed for bigger dataset

import pandas as pd

# Load in chunks of 100,000 rows
chunks = pd.read_csv("big_data.csv", chunksize=100000)

for chunk in chunks:
    # process chunk here
    print(chunk.shape)


# For numpy binary files (numerical data, faster to load than CSV)

import numpy as np

X = np.load("big_data.npy", mmap_mode="r")  # memory-mapped, avoids loading full file


# For HDF5 files (can handle very large arrays)

import h5py

with h5py.File("big_data.h5", "r") as f:
    print(list(f.keys()))   # see datasets
    X = f["dataset_name"]   # this is a reference, not yet loaded
    print(X.shape)
    # You can load slices without reading the whole dataset
    subset = X[:1000, :]    # load only first 1000 rows
