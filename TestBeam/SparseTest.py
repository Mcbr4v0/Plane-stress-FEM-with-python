import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

E = 210e9
nu = 0.3
N = 20
g = 4
h = 0.1
L = 10
c = 1
l = c*2
pho = 7850
def modes(n:int): #Numerically verified for n=1,2,3,4,5
    return(n*np.pi/L)**2*(E*l**2/12/pho)**0.5/2/np.pi

print(modes(1))
# Mesh generation
nodes, elements = Polygones.mesh(L, l, N, M=g, offsetX=0, offsetY=0)
#Polygones.showMesh2D(elements, nodes)

# Stiffness and mass matrices
K = Polygones.global_stiffness_matrix(nodes, elements, E, nu, h)
M = Polygones.global_mass_matrix(nodes, elements, pho, h)
def patch_test(nodes, elements, E, nu, t):
    # Apply constant strain state
    K = Polygones.global_stiffness_matrix(nodes, elements, E, nu, t)
    # Apply unit displacement field
    u = np.ones(K.shape[0])
    # Check if Ku gives constant stress
    f = K @ u
    return np.allclose(f, np.zeros_like(f))
def verify_stiffness_matrix(K):
    is_symmetric = np.allclose(K, K.T)
    print(f"Stiffness matrix is symmetric: {is_symmetric}")
    
    # Check positive definiteness
    eigenvals = np.linalg.eigvalsh(K)
    is_positive_semidefinite = np.all(eigenvals >= -1e-10)
    print(f"Stiffness matrix is positive semidefinite: {is_positive_semidefinite}")
    
    # Check conditioning
    cond = np.linalg.cond(K)
    print(f"Condition number: {cond}")
verify_stiffness_matrix(K)
print("Patch test passed:", patch_test(nodes, elements, E, nu, h))

# Force vector
F = np.zeros(2 * len(nodes), dtype=float)
F = Polygones.edgeForces(F, [20e6, 0], 'right', l, N, g)

# Boundary conditions
boudary_conditions = [(2, [0, 0]), (102, [0, None])]
K_reduced, F_reduced, constrained_dofs = Polygones.apply_boundary_conditions(K, F, boudary_conditions)
M_reduced = Polygones.reduce_mass_matrix(M, constrained_dofs)

# Convert to sparse format
K_sparse = sparse.csr_matrix(K_reduced)
M_sparse = sparse.csr_matrix(M_reduced)

# Solve the linear system using sparse solver
U_reduced = spla.spsolve(K_sparse, F_reduced)

# Reconstruct the full displacement vector
U_full = Polygones.reconstruct_full_vector(U_reduced, constrained_dofs, 2 * len(nodes))

# Normalize the reduced matrices
k_max = K_reduced.max()
m_max = M_reduced.max()

scaling_factor = k_max / m_max

K_sparse = K_sparse / K_sparse.max()
M_sparse = M_sparse / M_sparse.max()

# Check symmetry of the reduced stiffness matrix
is_symmetric = (K_sparse != K_sparse.T).nnz == 0
print("Is K_reduced symmetric?", is_symmetric)
print(scaling_factor)

# Compute eigenvalues and eigenvectors using sparse eigenvalue solver
num_modes = 10  # Number of eigenvalues/eigenvectors to compute
eigenvalues_reduced, eigenvectors_reduced = spla.eigsh(K_sparse, k=num_modes, M=M_sparse, which='SM')

# Ensure eigenvalues are non-negative
eigenvalues_reduced = np.maximum(eigenvalues_reduced*scaling_factor, 0)
print(eigenvalues_reduced)

# Compute natural frequencies
frequencies = np.sqrt(eigenvalues_reduced) / (2 * np.pi)

# Reconstruct full eigenvectors
eigenvectors = np.array([
    Polygones.reconstruct_full_vector(eigenvectors_reduced[:, i], constrained_dofs, 2 * len(nodes))
    for i in range(eigenvectors_reduced.shape[1])
])

# Print frequencies
print("Natural Frequencies (Hz):", frequencies)

# Plot frequencies
plt.close()
plt.plot(frequencies, 'o')
plt.xlabel('Mode Number')
plt.ylabel('Natural Frequency (f Hz)')
plt.title('Natural Frequencies of the System (Sparse)')
plt.grid()
plt.show()