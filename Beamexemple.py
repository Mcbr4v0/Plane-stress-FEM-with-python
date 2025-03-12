import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import pandas as pd
E = 1e3
M = 4
N = 20
t = 1
c = 10
L = 100
P=80
nu = 0.25
nodes,elements = Polygones.mesh(2*c,L,N,M,offsetX=0,offsetY=c)
Polygones.showMesh2D(elements,nodes)
K = Polygones.global_stiffness_matrix(nodes,elements,E,nu,t)
boudary_conditions = Polygones.boundry('right',[0,0],N,M)
N = Polygones.shapeFunction(nodes,elements,0)
print(N(-10,0))