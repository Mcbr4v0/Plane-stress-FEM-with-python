import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp
import pandas as pd
E = 210e9
nu = 0.3
N = 20
g = 4
t = 0.1
L = 10
c = 1
pho = 7850
nodes,elements = Polygones.mesh(2*c,L,N,M=g,offsetX=0,offsetY=0)
#Polygones.showMesh2D(elements,nodes)
K = Polygones.global_stiffness_matrix(nodes,elements,E,nu,t)
F  = np.zeros(2 * len(nodes), dtype=float)
F = Polygones.edgeForces(F,[20e6, 0],'right',2*c,N,g)
M = Polygones.global_mass_matrix(nodes,elements,pho,t)
boudary_conditions = Polygones.boundry('left',[0,0],N,g)
K_reduced, F_reduced,constrained_dofs = Polygones.apply_boundary_conditions(K, F, boudary_conditions)
M_reduced = Polygones.reduce_mass_matrix(M,constrained_dofs)
U_reduced =sp.solve(K_reduced, F_reduced)
U_full = Polygones.reconstruct_full_vector(U_reduced, constrained_dofs, 2 * len(nodes))
df = pd.DataFrame(U_full)
df.to_csv("displacement_vector.csv", index=False, header=False)
df = pd.DataFrame(K)
df.to_csv("stiffness_matrix.csv", index=False, header=False)
Ener = Polygones.globalEnergy(U_full,K)
deformed_nodes = Polygones.displacement(nodes, U_full,scale =1)
#Polygones.showDeform2D(elements,nodes,deformed_nodes)


alpha = 1e-10
A = K + alpha * M
eigenvalues, eigenvectors = sp.eigh(A, M)
omega_squared = eigenvalues - alpha
frequencies = np.sqrt(np.maximum(omega_squared, 0))/2/np.pi
print(frequencies)


Polygones.animate_plate_oscillation(nodes,elements,eigenvectors,frequencies,mode = 1,timeScale=1,scaleTot=2)

plt.close()
plt.plot(frequencies/2/np.pi, 'o')
plt.xlabel('Mode Number')
plt.ylabel('Natural Frequency (f Hz)')
#plt.yscale('log')
plt.title('Natural Frequencies of the System')
plt.grid()
plt.show()
