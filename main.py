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



E = 210e9
nu = 0.3
N = 20
G = 4
t = 0.1
L = 10
l = 2
pho = 7850
elt = 'quad' 
met = 'fine'

I = l*t**3/12   #moment of inertia 
A = l*t         #area of the cross section

mode = []
def modes(n:int): #Numerically verified for n=1,2,3,4,5
    return(n*np.pi/L)**2*(E*l**2/12/pho)**0.5/2/np.pi

for i in range(1,8):
    mode.append(modes(i))
    print(f"Mode number {i}",modes(i))

nodes,elements = Polygones.mesh(L,l,N,G,element_type=elt,mesh_type=met,offsetY = l/2)

Polygones.showMesh2D(nodes,elements,element_type=elt,mesh_type=met,show=False)

K = pd.read_csv('stiffness_matrix.txt', sep='\t', header=None).to_numpy()
M = pd.read_csv('mass_matrix.txt', sep='\t', header=None).to_numpy()
F = np.zeros(2*len(nodes),dtype=float)


boudary_conditions = [(4, [0, 0]), (355, [None, 0]),(184,[0,None]),(175,[0,None])]
K_reduced,F_reduced,constrained_dofs = Polygones.apply_boundary_conditions(K,F,boudary_conditions)
M_reduced = Polygones.reduce_mass_matrix(M,constrained_dofs)

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
print("Condition number of K_reduced:", cond_K)
print("Condition number of M_reduced:", cond_M)

eigenvalues_dense, eigenvectors_dense = scp.eigh(K_reduced, M_reduced)
eigenvalues_dense = np.maximum(eigenvalues_dense, 0)  # Ensure non-negative eigenvalues
frequencies_dense = np.sqrt(eigenvalues_dense * scaling_factor) / (2 * np.pi)
eigenvectors_reconstructed = Polygones.reconstruct_full_vectors(eigenvectors_dense, constrained_dofs, 2 * len(nodes))

for i in range(20):
    Polygones.animate_plate_oscillation(nodes,elements,eigenvectors_reconstructed,frequencies_dense,mode = i,element_type=elt,mesh_type=met,save_animation=True,filename=f"animations/mode_{i+1}.gif")

