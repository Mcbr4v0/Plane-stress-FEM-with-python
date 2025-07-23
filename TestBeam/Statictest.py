import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scp
import pandas as pd
from scipy.sparse.linalg import eigsh
import sympy as sp
import time
'''
This script performs a static analysis of a beam structure using finite element methods.
It generates meshes for triangular and quadrilateral elements, computes the global stiffness and mass matrices,
The load is applied trhough out the beam, and boundary conditions are applied to simulate fixed supports.
'''
# Material and geometric properties
E = 210e9
nu = 0.3
t = 0.1
l = 2
L = 20
P = 1e4
met = 'coarse'

# Lists to store results
G_values = [2, 4, 6]
element_types = ['tri', 'quad']
results = {
    'tri': {'max_disp': [], 'expected': [], 'diff': [], 'time': []},
    'quad': {'max_disp': [], 'expected': [], 'diff': [], 'time': []}
}

# Create figure with 2 rows (tri/quad) and 3 columns (G values)
plt.figure(figsize=(15, 10))

for elt_idx, elt in enumerate(element_types):
    print(f"\n{'='*50}")
    print(f"Testing {elt.upper()} elements")
    print('='*50)
    
    for idx, G in enumerate(G_values):
        print(f"\nRunning simulation for G = {G}")
        
        # Update mesh parameters
        N = 10*G
        I = t*l**3/12
        exceptedDisplacement = P*L**4/(E*I)*5/384
        
        # Generate mesh
        start_time = time.time()
        nodes, elements = Polygones.mesh(L, l, N, G, element_type=elt, mesh_type=met, offsetY=l/2)
        
        # Create subplot
        plt.subplot(2, 3, elt_idx*3 + idx + 1)
        Polygones.showMesh2D(nodes, elements, element_type=elt, mesh_type=met, show=False)
        
        # Compute stiffness matrix
        K = Polygones.global_stiffness_matrix(nodes, elements, E, nu, t, elt, met, integration='gauss')
        
        # Create force vector
        F = np.zeros(2*len(nodes))
        for i in range(len(nodes)):
            F[2*i+1] = -P*L/len(nodes)
        
        # Apply boundary conditions
        node1, node2 = int(G/2), int((len(nodes)-1)-G/2)
        boudary_conditions = [(node1, [0, 0]), (node2, [None, 0])]
        K_reduced, F_reduced, constrained_dofs = Polygones.apply_boundary_conditions(K, F, boudary_conditions)
        
        # Solve system
        U_reduced = np.linalg.solve(K_reduced, F_reduced)
        computation_time = time.time() - start_time
        
        # Reconstruct full displacement vector
        U = np.zeros(2*len(nodes))
        U[constrained_dofs] = 0
        U[np.setdiff1d(np.arange(len(U)), constrained_dofs)] = U_reduced
        
        # Store results
        max_disp = np.max(np.abs(U))
        results[elt]['max_disp'].append(max_disp)
        results[elt]['expected'].append(exceptedDisplacement)
        results[elt]['diff'].append(abs((max_disp - exceptedDisplacement)*100/exceptedDisplacement))
        results[elt]['time'].append(computation_time)
        
        # Display deformed shape
        deformed_nodes = Polygones.displacement(nodes, U)
        Polygones.showDeform2D(nodes, deformed_nodes, elements, element_type=elt, mesh_type=met, show=False)
        plt.title(f'{elt.upper()}, G = {G}\nMax Disp: {max_disp:.2e}\nTime: {computation_time:.2f}s')

plt.tight_layout()
plt.show()

# Print summary tables
for elt in element_types:
    print(f"\n{elt.upper()} Elements Results:")
    print("-" * 90)
    print(f"{'G':^10}{'Max Disp':^20}{'Expected':^20}{'Difference %':^15}{'Time (s)':^15}")
    print("-" * 90)
    for i, G in enumerate(G_values):
        print(f"{G:^10}{results[elt]['max_disp'][i]:^20.6e}"
              f"{results[elt]['expected'][i]:^20.6e}"
              f"{results[elt]['diff'][i]:^15.2f}{results[elt]['time'][i]:^15.2f}")
    print("-" * 90)

# Plot convergence comparison
plt.figure(figsize=(10, 6))
for elt in element_types:
    plt.plot(G_values, results[elt]['diff'], 'o-', label=f'{elt.upper()} Elements')
plt.xlabel('Number of elements in Y direction (G)')
plt.ylabel('Error percentage')
plt.title('Convergence Study: TRI vs QUAD')
plt.grid(True)
plt.legend()
plt.show()

# Plot computation time comparison
plt.figure(figsize=(10, 6))
for elt in element_types:
    plt.plot(G_values, results[elt]['time'], 'o-', label=f'{elt.upper()} Elements')
plt.xlabel('Number of elements in Y direction (G)')
plt.ylabel('Computation Time (s)')
plt.title('Performance Comparison: TRI vs QUAD')
plt.grid(True)
plt.legend()
plt.show()