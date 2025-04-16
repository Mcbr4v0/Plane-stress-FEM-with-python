import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
#constants
E = 210e9
nu = 0.3
N = 40 # Number of elements in the x-direction
g = 8# Number of elements in the y-direction
h = 0.1
L = 10
c = 1
l = c*2
pho = 7850
mode = []
def modes(n:int): #Numerically verified for n=1,2,3,4,5
    return(n*np.pi/L)**2*(E*l**2/12/pho)**0.5/2/np.pi
for i in range(1,8):
    mode.append(modes(i))
    print(f"Mode number {i}",modes(i))


# Mesh generation
nodes, elements = Polygones.mesh(L, l, N, M=g, offsetX=0, offsetY=c)
#Polygones.showMesh2D(elements, nodes)

# Stiffness and mass matrices
K = Polygones.global_stiffness_matrix(nodes, elements, E, nu, h)
M = Polygones.global_mass_matrix(nodes, elements, pho, h)
#plt.spy(K)
#plt.title("Sparsity Pattern of Global Stiffness Matrix")
#plt.show()
is_symmetric = np.allclose(K, K.T)
print("Is global stiffness matrix symmetric?", is_symmetric)
# Force vector
F = np.zeros(2 * len(nodes), dtype=float)
F = Polygones.edgeForces(F, [20e6, 0], 'right', l, N, g)
df = pd.DataFrame(K)
df.to_csv("stiffness_matrix.txt",index = False,header=False)
# Boundary conditions
boudary_conditions = [(4, [0, 0]), (364, [None, 0])]
#boudary_conditions =[(0, [0, 0]),[72,(0,None)], (360, [None, 0])]  [184,(0,None)],

K_reduced, F_reduced, constrained_dofs = Polygones.apply_boundary_conditions(K, F, boudary_conditions)
K_reconstruced = Polygones.reconstruct_full_matrix(K_reduced,constrained_dofs,2 * len(nodes))
M_reduced = Polygones.reduce_mass_matrix(M,constrained_dofs)

eigenvalues = np.linalg.eigvalsh(K_reduced)
is_positive_definite = np.all(eigenvalues > 0)
print("Is global stiffness matrix positive definite?", is_positive_definite)

# Solve the linear system using sparse solver
#U_reduced = spla.spsolve(K_sparse, F_reduced)

# Reconstruct the full displacement vector
#U_full = Polygones.reconstruct_full_vector(U_reduced, constrained_dofs, 2 * len(nodes))

# Normalize the reduced matrices
# Regularize matrices
'''
epsilon = 1
K_reduced += epsilon * np.eye(K_reduced.shape[0])
M_reduced += epsilon * np.eye(M_reduced.shape[0])
'''
# Normalize matrices
k_max = np.max(np.abs(np.diag(K_reduced)))
m_max = np.max(np.abs(np.diag(M_reduced)))
scaling_factor = k_max / m_max

K_reduced = K_reduced / k_max
M_reduced = M_reduced / m_max

# Check condition numbers
cond_K = np.linalg.cond(K_reduced)
cond_M = np.linalg.cond(M_reduced)
print("Condition number of K_reduced:", cond_K)
print("Condition number of M_reduced:", cond_M)

# Ensure symmetry
K_reduced = 0.5 * (K_reduced + K_reduced.T)
M_reduced = 0.5 * (M_reduced + M_reduced.T)

K_sparse = sparse.csr_matrix(K_reduced)
M_sparse = sparse.csr_matrix(M_reduced)

# Solve eigenvalue problem
num_modes = 10 # Number of eigenvalues/eigenvectors to compute
eigenvalues_reduced, eigenvectors = spla.eigsh(K_sparse, k=num_modes, M=M_sparse, which='SM')

# Don't scale the eigenvalues
frequencies = np.sqrt(np.abs(eigenvalues_reduced)) / (2 * np.pi)  # Convert to Hz

# Normalize eigenvectors with respect to mass matrix
for i in range(num_modes):
    norm = np.sqrt(np.abs(eigenvectors[:, i].T @ M_reduced @ eigenvectors[:, i]))
    eigenvectors[:, i] = eigenvectors[:, i] / norm

eigenvectors_reconstructed = Polygones.reconstruct_full_vectors(eigenvectors, constrained_dofs, 2 * len(nodes))


eigenvalues_reduced = np.real(eigenvalues_reduced)
print("Eigenvalues (reduced):", eigenvalues_reduced)

# Ensure eigenvalues are non-negative
eigenvalues_reduced = np.maximum(eigenvalues_reduced, 0)

# Verify K-Orthogonality
K_orthogonality = eigenvectors.T @ K_reduced @ eigenvectors
#print("K-Orthogonality Matrix (U^T K U):")
#print(K_orthogonality)

# Check if K-Orthogonality Matrix is diagonal
is_diagonal = np.allclose(K_orthogonality, np.diag(np.diagonal(K_orthogonality)))
print("Is K-Orthogonality Matrix Diagonal?", is_diagonal)
is_symmetric_K = np.allclose(K_reduced, K_reduced.T)
is_symmetric_M = np.allclose(M_reduced, M_reduced.T)
print("Is K_reduced symmetric?", is_symmetric_K)
print("Is M_reduced symmetric?", is_symmetric_M)
# If not diagonal, enforce K-Orthogonality
if not is_diagonal:
    # Diagonalize U^T K U
    eigvals, eigvecs = sp.eigh(K_orthogonality)

    # Transform eigenvectors to enforce K-orthogonality
    eigenvectors_k_orthogonal = eigenvectors @ eigvecs

    # Verify K-Orthogonality after correction
    K_orthogonality_corrected = eigenvectors_k_orthogonal.T @ K_reduced @ eigenvectors_k_orthogonal
    print("K-Orthogonality Matrix After Correction (U^T K U):")
    print(K_orthogonality_corrected)

    # Check again if it is diagonal
    is_diagonal_corrected = np.allclose(K_orthogonality_corrected, np.diag(np.diagonal(K_orthogonality_corrected)))
    print("Is K-Orthogonality Matrix Diagonal After Correction?", is_diagonal_corrected)

eigenvalues_dense, eigenvectors_dense = sp.eigh(K_reduced, M_reduced)
eigenvalues_dense = np.maximum(eigenvalues_dense, 0)  # Ensure non-negative eigenvalues
frequencies_dense = np.sqrt(eigenvalues_dense * scaling_factor) / (2 * np.pi)
eigenvectors_reconstructed = Polygones.reconstruct_full_vectors(eigenvectors_dense, constrained_dofs, 2 * len(nodes))

for i in range(5):
    Polygones.animate_plate_oscillation(nodes,elements,eigenvectors_reconstructed,frequencies_dense,mode = i)


print("\nComparison with analytical modes:")
for i in range(min(len(mode), len(frequencies_dense))):
    diff = abs(frequencies_dense[i] - mode[i])
    print(f"Mode {i+1}: Numerical = {frequencies_dense[i]:.2f} Hz, Analytical = {mode[i]:.2f} Hz, Difference = {diff:.2f} Hz")
# Plot corrected frequencies
plt.close()
plt.plot(frequencies_dense, 'o')
plt.xlabel('Mode Number')
plt.ylabel('Natural Frequency (f Hz)')
plt.title('Corrected Natural Frequencies of the System')
plt.grid()
plt.show()