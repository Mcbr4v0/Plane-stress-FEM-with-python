import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
'''
This script demonstrates the use of the Polygones module to compute the global stiffness matrix for a 2D mesh,
aApply boundary conditions, and solve the resulting linear system using both `np.linalg.solve` and `scipy.linalg.solve_banded`.
It skips the creation of a mesh and boundary conditions, focusing instead on the stiffness matrix and solving the system.
It also shows how to visualize the deformed shape of the mesh after applying a force.
'''
nodes = np.array([[0,0],[1,0],[2,0],[2,1],[1,1],[0,1]],dtype = float)
elements = {
    0: [0,1,4],
    1: [4,1,2],
    2: [2,3,4],
    3: [4,5,0], 
    }
E = 200e9
nu = 0.3
t = 0.1
K = Polygones.global_stiffness_matrix(nodes,elements,E,nu,t)
boundary_conditions = [(0, [0,0]), (5, [0,0])]
F = np.zeros(2 * len(nodes), dtype=float)  # Ensure F is of type float
F[8] = 1e10  # Apply force at node 4 in x-direction
F[9] = 1e10  # Apply force at node 4 in y-direction

# Apply boundary conditions
K, F,constrainedDofs = Polygones.apply_boundary_conditions(K, F, boundary_conditions)

# Convert the global stiffness matrix to banded format
lower_bandwidth = 2  # Number of sub-diagonals
upper_bandwidth = 2  # Number of super-diagonals
K_banded = Polygones.convert_to_banded(K, lower_bandwidth, upper_bandwidth)

# Solve the system using scipy.linalg.solve_banded
Ureduced = scipy.linalg.solve_banded((lower_bandwidth, upper_bandwidth), K_banded, F)
U = Polygones.reconstruct_full_vector(Ureduced, constrainedDofs,len(nodes)*2)
print(U)
deformed_nodes = Polygones.displacement(nodes ,U)
N = Polygones.shapeFunction(nodes,elements,1)
print(N(1,0))
Polygones.showDeform2D(nodes, deformed_nodes, elements)