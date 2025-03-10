import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
nodes = np.array([[0,0],[1,0],[2,0],[2,1],[1,1],[0,1]],dtype = float)
elements = {
    1: [0,1,4],
    2: [4,1,2],
    3: [2,3,4],
    4: [4,5,0], 
    }
E = 200e9
nu = 0.3
K = Polygones.global_stiffness_matrix(nodes,elements,E,nu)
boundary_conditions = [(0, [0,0]), (5, [0,0])]
F = np.zeros(2 * len(nodes), dtype=float)  # Ensure F is of type float
F[8] = 1e10  # Apply force at node 4 in x-direction
F[9] = 1e10  # Apply force at node 4 in y-direction

# Apply boundary conditions
K, F = Polygones.apply_boundary_conditions(K, F, boundary_conditions)

# Convert the global stiffness matrix to banded format
lower_bandwidth = 2  # Number of sub-diagonals
upper_bandwidth = 2  # Number of super-diagonals
K_banded = Polygones.convert_to_banded(K, lower_bandwidth, upper_bandwidth)

# Solve the system using scipy.linalg.solve_banded
U = scipy.linalg.solve_banded((lower_bandwidth, upper_bandwidth), K_banded, F)
deformed_nodes = Polygones.displacement(nodes ,U)
Polygones.showDeform2D(elements,nodes,deformed_nodes)