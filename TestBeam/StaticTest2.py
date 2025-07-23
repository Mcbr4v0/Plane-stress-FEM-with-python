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
The load is applied at the center of the beam, and boundary conditions are applied to simulate fixed supports.
'''

# Material and geometric properties
E = 210e9
nu = 0.3
N = 10
t = 0.1
l = 2
L = 20
pho = 7850
P = 1e4
I = t*l**3/12   # moment of inertia 
met = 'coarse'

# Test parameters
G_values = [2, 4, 6]
element_types = ['tri', 'quad']
results = {elt: {G: {} for G in G_values} for elt in element_types}

# Create figure with 2 rows (tri/quad) and 3 columns (G values)
plt.figure(figsize=(15, 10))

for elt_idx, elt in enumerate(element_types):
    print(f"\n{'='*50}")
    print(f"Testing {elt.upper()} elements")
    print('='*50)
    
    for G_idx, G in enumerate(G_values):
        print(f"\nTesting G = {G}")
        start_time = time.time()
        
        # Expected displacement for center load
        exceptedDisplacement = P*L**4/(48*E*I)
        
        # Create mesh
        nodes, elements = Polygones.mesh(L, l, N, G, element_type=elt, mesh_type=met, offsetY=l/2)
        
        # Create subplot
        plt.subplot(2, 3, elt_idx*3 + G_idx + 1)
        
        # Compute stiffness matrix
        K = Polygones.global_stiffness_matrix(nodes, elements, E, nu, t, elt, met, integration='gauss')
        
        # Initialize force vector
        F = np.zeros(2*len(nodes))
        
        # Apply concentrated load at center node
        center_node = int(len(nodes)/2)
        F[2*center_node + 1] = -P*L
        
        # Define boundary conditions
        node1, node2 = int(G/2), int((len(nodes)-1)-G/2)
        boudary_conditions = [(node1, [0, 0]), (node2, [None, 0])]
        
        # Apply boundary conditions
        K_reduced, F_reduced, constrained_dofs = Polygones.apply_boundary_conditions(K, F, boudary_conditions)
        
        # Solve system
        U_reduced = np.linalg.solve(K_reduced, F_reduced)
        computation_time = time.time() - start_time
        
        # Reconstruct full displacement vector
        U = np.zeros(2*len(nodes))
        U[constrained_dofs] = 0
        U[np.setdiff1d(np.arange(len(U)), constrained_dofs)] = U_reduced
        
        # Check solution quality
        residual = np.linalg.norm(K_reduced @ U_reduced - F_reduced)
        max_displacement = np.max(np.abs(U))
        difference = abs((max_displacement - abs(exceptedDisplacement))*100/exceptedDisplacement)
        
        # Store results
        results[elt][G] = {
            'max_disp': max_displacement,
            'residual': residual,
            'difference': difference,
            'time': computation_time
        }
        
        # Plot deformed shape
        deformed_nodes = Polygones.displacement(nodes, U)
        Polygones.showDeform2D(nodes, deformed_nodes, elements, element_type=elt, mesh_type=met, show=False)
        plt.title(f'{elt.upper()}, G={G}\nMax Disp: {max_displacement:.2e}\nTime: {computation_time:.2f}s')

plt.tight_layout()
plt.show()

# Print comparison table
print("\nResults Comparison:")
print("-" * 110)
print(f"{'Element':^10}{'G':^5}{'Max Disp':^20}{'Expected':^20}{'Difference %':^15}{'Residual':^15}{'Time (s)':^15}")
print("-" * 110)
for elt in element_types:
    for G in G_values:
        print(f"{elt:^10}{G:^5}{results[elt][G]['max_disp']:^20.6e}{exceptedDisplacement:^20.6e}"
              f"{results[elt][G]['difference']:^15.2f}{results[elt][G]['residual']:^15.2e}"
              f"{results[elt][G]['time']:^15.2f}")
print("-" * 110)

# Plot convergence comparison
plt.figure(figsize=(10, 6))
for elt in element_types:
    plt.plot(G_values, [results[elt][G]['difference'] for G in G_values], 'o-', label=f'{elt.upper()}')
plt.xlabel('Number of elements in Y direction (G)')
plt.ylabel('Error percentage')
plt.title('Convergence Study: TRI vs QUAD')
plt.grid(True)
plt.legend()
plt.show()