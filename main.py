import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp
import pandas as pd
from scipy.sparse.linalg import eigsh

E = 210e9
nu = 0.3
N = 10
t = 0.01
L = 2
l = 4
pho = 7850
nodes,elements = Polygones.mesh(L,L,N)

#Polygones.showMesh2D(elements,nodes)


print(Polygones.shapeFunctionCoeff(nodes,elements,0))
K = Polygones.global_stiffness_matrix(nodes, elements, E, nu,t)
M = Polygones.global_mass_matrix(nodes, elements,pho,t)
F = np.zeros(2 * len(nodes), dtype=float)
F = Polygones.edgeForces(F, [20e6, 0], 'right',L,N)
boundary_conditions = Polygones.boundry('left', [0, 0],N)
K_reduced, F_reduced,constrained_dofs = Polygones.apply_boundary_conditions(K, F, boundary_conditions)
U = sp.solve(K_reduced, F_reduced)
U_full = Polygones.reconstruct_full_vector(U, constrained_dofs, 2 * len(nodes))
deformed_nodes = Polygones.displacement(nodes, U_full,scale =1)
Polygones.showDeform2D(elements,nodes,deformed_nodes)

# Save the stiffness matrix (K) to a CSV file
K_df = pd.DataFrame(K)  # Convert the matrix to a DataFrame
K_df.to_csv("stiffness_matrix.csv", index=False, header=False)  # Save to CSV without row/column headers

# Save the mass matrix (M) to a CSV file
M_df = pd.DataFrame(M)  # Convert the matrix to a DataFrame
M_df.to_csv("mass_matrix.csv", index=False, header=False)  # Save to CSV without row/column headers

# Save the displacement vector (U) to a CSV file
U_df = pd.DataFrame(U_full)  # Convert the matrix to a DataFrame
U_df.to_csv("displacement_vector.csv", index=False, header=False)  # Save to CSV without row/column headers

M_reduced = Polygones.reduce_mass_matrix(M, constrained_dofs)

eigenvalues_reduced, eigenvectors_reduced = sp.eigh(K_reduced, M_reduced)

# Compute natural frequencies (ω) from eigenvalues (ω²)
frequencies = np.sqrt(eigenvalues_reduced)/2/np.pi
eigenvectors = Polygones.reconstruct_full_vectors(eigenvectors_reduced, constrained_dofs, 2 * len(nodes))
'''
# Create a header for the CSV file
num_modes = eigenvectors_reduced.shape[1]
header = ["Eigenvalue"] + [f"DOF {i+1} " for i in range(num_modes)]

# Combine eigenvalues and eigenvectors into a single matrix
eigen_data = np.hstack((frequencies.reshape(-1, 1), eigenvectors_reduced))

# Save the combined data to a CSV file with a descriptive header
eigen_data_df = pd.DataFrame(eigen_data, columns=header)  # Add the header to the DataFrame
eigen_data_df.to_csv("eigenvalues_eigenvectors.csv", index=False)  # Save to CSV
# Combine eigenvalues and eigenvectors into a single matrix

# Save the combined data to a CSV file with a descriptive header
#eigenvectorsdf = pd.DataFrame(eigenvectors)  # Add the header to the DataFrame
#eigenvectorsdf.to_csv("eigenvectors.csv", index=False)  # Save to CSV



Polygones.animate_plate_oscillation(nodes,elements,eigenvectors,frequencies,mode = 0,timeScale=1)

print("Condition number of K:", np.linalg.cond(K))
print("Condition number of M:", np.linalg.cond(M))
print("Constrained DOFs:", constrained_dofs)
print(frequencies)
# Print the results
plt.close()
plt.plot(frequencies, 'o')
plt.xlabel('Mode Number')
plt.ylabel('Natural Frequency (f Hz)')
plt.title('Natural Frequencies of the System')
plt.grid()
plt.show()
'''
K_normalised,maxKvalue = Polygones.normalize_matrix(K)
M_normalised,maxMvalue = Polygones.normalize_matrix(M)
eigenvalues, eigenvectors = sp.eigh(K_normalised, M_normalised)

frequencies = np.sqrt(eigenvalues*maxKvalue)/2/np.pi
#print(frequencies)


alpha = 1e5
A = K + alpha * M
eigenvalues, eigenvectors = sp.eigh(A, M)
omega_squared = eigenvalues - alpha
frequenciescomputed = np.sqrt(np.maximum(omega_squared, 0))/2/np.pi
print(frequenciescomputed)

#compare the two methods

Polygones.animate_plate_oscillation(nodes,elements,eigenvectors,frequencies,mode = 0,timeScale=10)

# Save the combined data to a CSV file with a descriptive header
#eigenvectorsdf = pd.DataFrame(eigenvectors)  # Add the header to the DataFrame
#eigenvectorsdf.to_csv("eigenvectorsfull.csv", index=False)  # Save to CSV

num_modes = 10  # Number of modes to compute
eigenvalues, eigenvectors = eigsh(K, M=M, k=num_modes, which='SA')

# Compute natural frequencies
frequencies = np.sqrt(np.maximum(eigenvalues, 0)) / (2 * np.pi)
print("First natural frequencies (Hz):", frequencies)

plt.close()
plt.plot(frequenciescomputed, 'o')
plt.xlabel('Mode Number')
plt.ylabel('Natural Frequency (f Hz)')
plt.title('Natural Frequencies of the System')
plt.grid()
plt.show()




