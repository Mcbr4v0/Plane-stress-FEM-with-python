import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import pandas as pd
E = 200e9
nu = 0.3
N = 10
t = 0.1
nodes,elements = Polygones.mesh(10,10,N)

bc1 = Polygones.boundry(N,'left',[0,0])
boundary_conditions = bc1
F = np.zeros(2 * len(nodes), dtype=float) 

l = 10/N
F= Polygones.forces(N,F,[1e11,0],'right',l)
K = Polygones.global_stiffness_matrix(nodes,elements,E,nu,t)
K,F = Polygones.apply_boundary_conditions(K,F,boundary_conditions)
lower_bandwidth,upper_bandwidth  = Polygones.test_bandiwth(K)
K_banded = Polygones.convert_to_banded(K, lower_bandwidth, upper_bandwidth)
Uprime = np.linalg.solve(K, F)

U = scipy.linalg.solve_banded((lower_bandwidth, upper_bandwidth), K_banded, F)
df = pd.DataFrame(K_banded)
df.to_csv('K_banded.csv',index=False)
df = pd.DataFrame(K)
df.to_csv('K.csv',index=False)
print(upper_bandwidth)
deformed_nodes = Polygones.displacement(nodes ,Uprime)  
Polygones.showDeform2D(elements,nodes,deformed_nodes)
deformed_nodes = Polygones.displacement(nodes ,U)  
Polygones.showDeform2D(elements,nodes,deformed_nodes)
