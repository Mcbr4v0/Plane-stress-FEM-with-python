import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import pandas as pd
E = 200e9
nu = 0.3
N = 30
t = 0.01
l = 10
nodes,elements = Polygones.mesh(l,l,N)

bc1 = Polygones.boundry('left',[0,0],N)
bc2 = Polygones.boundry('right',[-1,0],N)
#boundary_conditions = Polygones.merge_boudary_conditions(bc1,bc2)
boundary_conditions = bc1
F = np.zeros(2 * len(nodes), dtype=float) 


F= Polygones.edgeForces(F,[1e7,0],'right',l,N)
K = Polygones.global_stiffness_matrix(nodes,elements,E,nu,t)
K,F = Polygones.apply_boundary_conditions(K,F,boundary_conditions)
lower_bandwidth,upper_bandwidth  = Polygones.test_bandiwth(K)
K_banded = Polygones.convert_to_banded(K, lower_bandwidth, upper_bandwidth)
Uprime = np.linalg.solve(K, F)

U = scipy.linalg.solve_banded((lower_bandwidth, upper_bandwidth), K_banded, F)
'''df = pd.DataFrame(K_banded)
df.to_csv('K_banded.csv',index=False)
df = pd.DataFrame(K)
df.to_csv('K.csv',index=False)'
'''
print(upper_bandwidth)
deformed_nodes = Polygones.displacement(nodes ,Uprime)  
Polygones.showDeform2D(elements,nodes,deformed_nodes)

deformed_nodes = Polygones.displacement(nodes ,U)  


