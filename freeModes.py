import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scp
import pandas as pd
from scipy.sparse.linalg import eigsh
import sympy as sp
import time
import os
if not os.path.exists('animations'):
    os.makedirs('animations')

'''
This script computes the global stiffness and mass matrices for a beam structure using finite element methods.
It generates a mesh, but imports the stiffness and mass matrices from text files.
Note: Depending on the script you are using, the mesh type and element type may vary.
Copy and paste mesh parameters and element parameters from the script you are using.
It creates animations of the first 10 modes of oscillation of the plate.
The animations are saved in the 'animations' directory.
The analytical frequencies of the first 7 modes are also printed for comparison.
'''


E = 210e9
nu = 0.3
N = 80
G = 8
t = 0.01
l = 2
L = 20
pho = 7850
P = 1e4
I = t*l**3/12   #moment of inertia 
elt = 'tri' 
met = 'coarse'

mode = []
def modes(n:int): #Numerically verified for n=1,2,3,4,5
    return(n/L)**2*np.pi/2*(E*l**2/12/pho)**0.5

for i in range(1,8):
    mode.append(modes(i))
    print(f"Mode number {i}",modes(i))

nodes,elements = Polygones.mesh(L,l,N,G,element_type=elt,mesh_type=met,offsetY = l/2)

Polygones.showMesh2D(nodes,elements,element_type=elt,mesh_type=met,show=True)

K = pd.read_csv('stiffness_matrix.txt', sep='\t', header=None).to_numpy()
M = pd.read_csv('mass_matrix.txt', sep='\t', header=None).to_numpy()
F = np.zeros(2*len(nodes),dtype=float)

node1,node2 = int(G/2),int((len(nodes)-1)-G/2)
boudary_conditions = [(node1, [0, 0]), (node2, [None, 0])]
K_reduced,F_reduced,constrained_dofs = Polygones.apply_boundary_conditions(K,F,boudary_conditions)
M_reduced = Polygones.reduce(M,constrained_dofs)

k_max = np.max(np.abs(np.diag(K_reduced)))
m_max = np.max(np.abs(np.diag(M_reduced)))
scaling_factor = k_max / m_max

K_reduced = K_reduced / k_max
M_reduced = M_reduced / m_max

eigenvalues = np.linalg.eigvalsh(K_reduced)
is_positive_definite = np.all(eigenvalues > 0)
print("Is global stiffness matrix positive definite?", is_positive_definite)

cond_K = np.linalg.cond(K_reduced)
cond_M = np.linalg.cond(M_reduced)
print(f"Condition number of K_reduced: {cond_K:.2e}")
print(f"Condition number of M_reduced: {cond_M:.2e}" )

eigenvalues_dense, eigenvectors_dense = scp.eigh(K_reduced, M_reduced)
eigenvalues_dense = np.maximum(eigenvalues_dense, 0)  # Ensure non-negative eigenvalues
frequencies_dense = np.sqrt(eigenvalues_dense * scaling_factor) / (2 * np.pi)
eigenvectors_reconstructed = Polygones.reconstruct_full_vectors(eigenvectors_dense, constrained_dofs, 2 * len(nodes))

print ("verify ortogonality",eigenvectors_reconstructed[1].T@K@eigenvectors_reconstructed[0])

for i in range(10):
   Polygones.animate_plate_oscillation(nodes,elements,eigenvectors_reconstructed,frequencies_dense,mode = i,element_type=elt,mesh_type=met,save_animation=False,filename=f"animations/coarseMode_{i+1}.gif")

