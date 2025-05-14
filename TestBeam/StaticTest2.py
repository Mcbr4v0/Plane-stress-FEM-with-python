import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scp
import pandas as pd
from scipy.sparse.linalg import eigsh
import sympy as sp
import time

# Material and geometric properties
E = 210e9
nu = 0.3
N = 40
G = 4
t = 0.01
l = 2
L = 20
pho = 7850
P = 1e4
I = t*l**3/12   # moment of inertia 
elt = 'quad' 
met = 'coarse'

# Expected displacement for center load
exceptedDisplacement = P*L**4/(48*E*I)

# Create mesh
nodes, elements = Polygones.mesh(L,l,N,G,element_type=elt,mesh_type=met,offsetY = l/2)

# Show mesh
Polygones.showMesh2D(nodes,elements,element_type=elt,mesh_type=met,show=False)

# Load stiffness matrix
K = pd.read_csv('stiffness_matrix.txt', sep='\t', header=None).to_numpy()

# Initialize force vector
F = np.zeros(2*len(nodes), dtype=float)

# Apply concentrated load at center node
center_node = int(len(nodes)/2)  # Assuming the center node is at the middle
F[2*center_node + 1] = -P*L  # Apply force in y-direction

# Define boundary conditions
node1, node2 = int(G/2), int((len(nodes)-1)-G/2)
boudary_conditions = [(node1, [0, 0]), (node2, [None, 0])]
print("Boundary conditions:", boudary_conditions)

# Apply boundary conditions
K_reduced, F_reduced, constrained_dofs = Polygones.apply_boundary_conditions(K, F, boudary_conditions)

# Solve system
U_reduced = np.linalg.solve(K_reduced, F_reduced)

# Reconstruct full displacement vector
U = np.zeros(2*len(nodes), dtype=float)
U[constrained_dofs] = 0
U[np.setdiff1d(np.arange(len(U)), constrained_dofs)] = U_reduced

# Check solution quality
residual = np.linalg.norm(K_reduced @ U_reduced - F_reduced)
print(f"Solution residual: {residual:.2e}")

# Plot deformed shape
deformed_nodes = Polygones.displacement(nodes, U)
Polygones.showDeform2D(nodes, deformed_nodes, elements, element_type=elt, mesh_type=met, show=True)

# Compare with analytical solution
max_displacement = np.max(np.abs(U))
print("Maximum displacement:", max_displacement)
print("Expected displacement:", exceptedDisplacement)
difference = abs((max_displacement - abs(exceptedDisplacement))*100/exceptedDisplacement)
print("Difference in percentage:", difference)