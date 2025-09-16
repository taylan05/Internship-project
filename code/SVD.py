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
#     display_name: Python (Internship Project)
#     language: python
#     name: internship_project
# ---

# +
import numpy as np

# Define a 3x2 matrix
A = np.array([[1, 1],
              [1, 0],
              [0, 1]])

print("Original matrix A:")
print(A)

# Compute SVD
U, S, Vt = np.linalg.svd(A)

print("\nU (left singular vectors):")
print(U)

print("\nSingular values (vector S):")
print(S)

print("\nV^T (right singular vectors transposed):")
print(Vt)

# Convert S (vector) to Sigma (3x2 diagonal matrix)
Sigma = np.zeros_like(A, dtype=float)
np.fill_diagonal(Sigma, S)

print("\nSigma (diagonal matrix of singular values):")
print(Sigma)

# Verify reconstruction: A ≈ U Σ V^T
A_reconstructed = U @ Sigma @ Vt

print("\nReconstructed A:")
print(A_reconstructed)

# Check the difference
print("\nDifference (A - A_reconstructed):")
print(A - A_reconstructed)


# +
import sympy as sp

# Define a 3x2 matrix with exact integers
B = sp.Matrix([[1, 1],
               [1, 0],
               [0, 1]])

print("Original matrix B:")
sp.pprint(B)

# Compute the SVD (SymPy's svd gives U, S, V^T)
U, S, Vt = B.singular_value_decomposition()

print("\nU (left singular vectors):")
sp.pprint(U)

print("\nSingular values (diagonal matrix S):")
sp.pprint(S)

print("\nV^T (right singular vectors transposed):")
sp.pprint(Vt)

# Verify reconstruction
B_reconstructed = U * S * Vt

print("\nReconstructed A:")
sp.pprint(B_reconstructed)

print("\nDifference (A - A_reconstructed):")
sp.pprint(B - B_reconstructed)


# +
import sympy as sp

# Define a 3x2 matrix
B = sp.Matrix([[1, 1],
               [1, 0],
               [0, 1]])

print("Original matrix A:")
sp.pprint(B)

# Step 1: Compact SVD
U_compact, S_compact, Vt = B.singular_value_decomposition()

print("\nCompact U (3x2):")
sp.pprint(U_compact)

print("\nCompact Σ (2x2):")
sp.pprint(S_compact)

print("\nV^T (2x2):")
sp.pprint(Vt)

# Step 2: Extend U to 3x3 using Gram-Schmidt
U_full = U_compact.copy()
# Pick a standard basis vector not in the span of U_compact
candidate = sp.Matrix([0, 0, 1])
# Orthogonalize against U_compact's columns
for col in U_compact.columnspace():
    candidate -= (col.dot(candidate)) * col
candidate = candidate / candidate.norm()  # normalize
U_full = U_full.row_join(candidate)

print("\nExtended U (3x3):")
sp.pprint(U_full)

# Step 3: Build Σ as 3x2
Sigma = sp.zeros(B.shape[0], B.shape[1])  # 3x2
for i in range(S_compact.shape[0]):
    Sigma[i, i] = S_compact[i, i]

print("\nΣ (3x2):")
sp.pprint(Sigma)

# Step 4: Verify reconstruction
B_reconstructed = U_full * Sigma * Vt

print("\nReconstructed B:")
sp.pprint(B_reconstructed)

print("\nDifference (B - B_reconstructed):")
sp.pprint(B - B_reconstructed)

# -


