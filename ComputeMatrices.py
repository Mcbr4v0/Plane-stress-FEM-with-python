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
N =20
G = 2
t = 0.01
l = 2
L = 20
pho = 7850
P = 1e4
I = t*l**3/12   #moment of inertia 
elt = 'quad' 
met = 'coarse'

mode = []
nodes,elements = Polygones.mesh(L,l,N,G,element_type=elt,mesh_type=met,offsetY = l/2)
tf = time.time()
K = Polygones.global_stiffness_matrix(nodes,elements,E,nu,t,elt,met,integration='gauss')
M = Polygones.global_mass_matrix(nodes,elements,pho,t,elt,met,integration='gauss')
dfK = pd.DataFrame(K)
dfK.to_csv('stiffness_matrix.txt', sep='\t', index=False, header=False)
dfM = pd.DataFrame(M)
dfM.to_csv('mass_matrix.txt', sep='\t', index=False, header=False)
tf = time.time() - tf
print("Time taken to compute the stiffness and mass matrices:", tf, "seconds")