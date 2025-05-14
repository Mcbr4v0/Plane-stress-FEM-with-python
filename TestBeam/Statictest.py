
import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scp
import pandas as pd
from scipy.sparse.linalg import eigsh
import sympy as sp
import time

E = 210e9
nu = 0.3
N =40
G = 4
t = 0.01
l = 2
L = 20
pho = 7850
P = 1e4
I = t*l**3/12   #moment of inertia 
elt = 'quad' 
met = 'coarse'
exceptedDisplacement =  P*L**4/(E*I)*5/384

nodes,elements = Polygones.mesh(L,l,N,G,element_type=elt,mesh_type=met,offsetY = l/2)

Polygones.showMesh2D(nodes,elements,element_type=elt,mesh_type=met,show=False)

K = pd.read_csv('stiffness_matrix.txt', sep='\t', header=None).to_numpy()
F = np.zeros(2*len(nodes),dtype=float)

for i in range(len(nodes)):
    F[2*i] = 0
    F[2*i+1] = -P*L/len(nodes)
node1,node2 = int(G/2),int((len(nodes)-1)-G/2)
boudary_conditions = [(node1, [0, 0]), (node2, [None, 0])]
print("Boundary conditions:", boudary_conditions)
K_reduced,F_reduced,constrained_dofs = Polygones.apply_boundary_conditions(K,F,boudary_conditions)

U_reduced = np.linalg.solve(K_reduced,F_reduced)
U = np.zeros(2*len(nodes),dtype=float)
U[constrained_dofs] = 0
U[np.setdiff1d(np.arange(len(U)), constrained_dofs)] = U_reduced

residual = np.linalg.norm(K_reduced @ U_reduced - F_reduced)
print(f"Solution residual: {residual:.2e}")

deformed_nodes = Polygones.displacement(nodes,U)
Polygones.showDeform2D(nodes,deformed_nodes,elements,element_type=elt,mesh_type=met,show=0)
max = np.max(np.abs(U))
print("Maximum displacement:",max)
print("Expected displacement at node 102:",exceptedDisplacement)
dif = abs((max - abs(exceptedDisplacement))*100/exceptedDisplacement)
print("Difference in percentage:",dif)