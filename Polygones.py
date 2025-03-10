""""
This program inted to solve the problem of a plate under load using the finite element method
The program is divided into two parts:
1- the first part is the definition of the functions that will be used in the second part
2- the second part is the implementation of the finite element method to solve the problem of a plate under load
This part is all function used later in the second part
showDeform2D: function to show the deformation of the plate
displacement: function to calculate the deformed nodes
L: function to calculate the Lagrange polynomial
Hermit_interpolation: function to calculate the hermite interpolation
shapeFunction: function to calculate the shape function
strainDisplacement: function to calculate the B matrix
local_stiffness_matrix: function to calculate the local stiffness matrix
global_stiffness_matrix: function to calculate the global stiffness matrix
apply_boundary_conditions: function to apply the boundary conditions
mesh: function to create the mesh
convert_to_banded: function to convert the global stiffness matrix to banded format
calculate_bandwith: function to calculate the bandwith of the matrix to remove the zeros
boundry: function to create the boundary conditions
merge_boudary_conditions: function to merge two boundary conditions
forces: function to apply the forces


Writen by: Arthur Boudehent
Last modification: 2025-04-03
"""

#imports
import matplotlib.pyplot as plt
import numpy as np

def showDeform2D(element,nodes,deformed_nodes):
    x,y,xdef,ydef = [],[],[],[]
    for i in element:
        for j in element[i]:#loop to get the coordinates of the nodes of the element
            x.append(nodes[j][0])
            y.append(nodes[j][1])#implement all nodes of the element
            xdef.append(deformed_nodes[j][0])
            ydef.append(deformed_nodes[j][1])
        x.append(nodes[element[i][0]][0]);x.append(nodes[element[i][2]][0])#close the triangle
        y.append(nodes[element[i][0]][1]);y.append(nodes[element[i][2]][1])
        xdef.append(deformed_nodes[element[i][0]][0]);xdef.append(deformed_nodes[element[i][2]][0])#close the triangle
        ydef.append(deformed_nodes[element[i][0]][1]);ydef.append(deformed_nodes[element[i][2]][1])
        
    plt.plot(x, y,marker='x', linestyle='dotted',color='blue',label='original')
    plt.plot(xdef, ydef,marker='x', linestyle='dotted',color='red',label='deformed')
    plt.axis('equal')
    plt.title("deformation of a plate under load")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

def displacement(nodes,U):  
    deformed_nodes = np.copy(nodes)
    for i in range(len(nodes)):
        deformed_nodes[i] = [nodes[i][0]+U[2*i],nodes[i][1]+U[2*i+1]]#adding the displacement to the original coordinates
    return deformed_nodes

def L(k, x_val,x):#Lagrange polynomial
        result = 1
        for i in range(len(x)):
            if i != k:
                result *= (x_val - x[i]) / (x[k] - x[i])
        return result  

def lagrange_interpolation(x, y):  #lagrange interpolation
      
    def P(x_val):
        result = 0
        for k in range(len(x)):
            result += y[k] * L(k, x_val,x)
        return result
    return P

def hermite_interpolation(x,y,yprime):#hermite interpolation
    n = len(x)
    if len(y) != n or len(yprime) != n:
        print("Error: the two arrays must have the same length")
        return None
    def P(x_val):
        result = 0
        for k in range(n):
            Lk = L(k, x_val,x)
            result += y[k] * Lk * Lk * (1 - 2 * Lk * (x_val - x[k])) + yprime[k] * Lk * Lk * (x_val - x[k])#i have replaced the derivative of L by L to simplify the calculation
        return result
    return P

def shapeFunction(nodes, element,nbElement):
    points = element[nbElement]
    L=[]
    for i in range(len(points)):# calulation of each coefficient of the shape function
        a = nodes[points[(i+1)%3]][0]*nodes[points[(i+2)%3]][1]-nodes[points[(i+2)%3]][1]*nodes[points[(i+1)%3]][0]
        b = nodes[points[(i+1)%3]][1]-nodes[points[(i+2)%3]][1]
        c = nodes[points[(i+2)%3]][0]-nodes[points[(i+1)%3]][0]
        L.append([a,b,c])
    return L

def strainDisplacement(nodes,elements,nbElement): #calcul of the B matrix, since the shape function is linear, the B matrix is constant
    points = elements[nbElement]
    L = shapeFunction(nodes,elements,nbElement)
    B = np.zeros((3,6))
    for i in range(len(points)):
        B[0,2*i] = L[i][1]
        B[0,2*i+1] = 0
        B[1,2*i] = 0
        B[1,2*i+1] = L[i][2]
        B[2,2*i] = L[i][2]
        B[2,2*i+1] = L[i][1]   
    return B

def local_stiffness_matrix(nodes,element,nbElement,E,nu,t):
    points = element[nbElement]
    Volume = 0.5*(nodes[points[0]][0]*(nodes[points[1]][1]-nodes[points[2]][1])
                +nodes[points[1]][0]*(nodes[points[2]][1]-nodes[points[0]][1])
                +nodes[points[2]][0]*(nodes[points[0]][1]-nodes[points[1]][1]))*t#calcul of the triangle volume
    B = strainDisplacement(nodes,element,nbElement)
    D = np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])*E/(1-nu**2)# to remove from the function because it depends on the shape of the element
    Ke = np.dot(B.T,np.dot(D,B))*Volume #since the B matrix is constant, the integral of B^TDB is equal to B^TDB*Volume
    return Ke

def global_stiffness_matrix(nodes,element,E,nu,t):
    n = len(nodes)
    K = np.zeros((2*n,2*n))
    for i in element:
        Ke = local_stiffness_matrix(nodes,element,i,E,nu,t)
        for j in range(3):
            for k in range(3):
                K[2*element[i][j],2*element[i][k]] += Ke[2*j,2*k]
                K[2*element[i][j],2*element[i][k]+1] += Ke[2*j,2*k+1]
                K[2*element[i][j]+1,2*element[i][k]] += Ke[2*j+1,2*k]
                K[2*element[i][j]+1,2*element[i][k]+1] += Ke[2*j+1,2*k+1]
    return K

def apply_boundary_conditions(K, F, bc):
    for node, displacement in bc:
        idx = 2 * node
        for i in range(len(K)):
            K[idx, i] = 0
            K[idx + 1, i] = 0
            K[i, idx] = 0
            K[i, idx + 1] = 0
        K[idx, idx] = 1
        K[idx + 1, idx + 1] = 1
        F[idx] = displacement[0]
        F[idx + 1] = displacement[1]
    return K, F

def mesh(L,l,N):
    n = N+1
    nodes = np.zeros((n**2,2))
    elements = {}
    for i in range(n):
        for j in range(n):
            nodes[i*n+j] = [i*l/(n-1),j*L/(n-1)]
    index = 0
    for i in range(n-1):
        for j in range(n-1):
            elements[index] = [j+i*n,j+i*n+n,j+i*n+1]
            index += 1
            elements[index] = [j+i*n+n+1,j+i*n+1,j+i*n+n]  
            index+=1        
    return nodes,elements

def convert_to_banded(K, lower_bandwidth, upper_bandwidth):
    n = K.shape[0]
    ab = np.zeros((lower_bandwidth + upper_bandwidth + 1, n))
    for i in range(n):
        for j in range(max(0, i - lower_bandwidth), min(n, i + upper_bandwidth + 1)):
            ab[upper_bandwidth + i - j, j] = K[i, j]
    return ab

def calculate_bandwith(K):
    n = K.shape[0]
    lower_bandwidth = 0
    upper_bandwidth = 0
    for i in range(n):
        for j in range(n):
            if K[i, j] != 0:
                if j < i:
                    lower_bandwidth = max(lower_bandwidth, i - j)
                else:
                    upper_bandwidth = max(upper_bandwidth, j - i)
    return lower_bandwidth, upper_bandwidth

def boundry(N,direct,disp):
    n = N+1
    boundry = []
    if direct == 'top':
        for i in range(n):
            boundry.append((i*n+n,disp))
    elif direct == 'bottom':
        for i in range(n):
            boundry.append((i*n,disp))
    elif direct == 'left':
        for i in range(n):
            boundry.append((i,disp))
    elif direct == 'right':
        for i in range(n):
            boundry.append((i+n*(n-1),disp))
    return boundry

def merge_boudary_conditions(bc1, bc2):
    bc = []
    for i in bc1:
        bc.append(i)
    for i in bc2:
        if i not in bc:
            bc.append(i)
    return bc

def forces(N,F,force,dir,l):
    n = N+1
    if dir == 'top':
        for i in range(n):
            F[2*(i*n+n-1)] += force[0]*l/(n**2)
            F[2*(i*n+n-1)+1] += force[1]*l/(n**2)
        for i in range(n):
            F[2*(i*n)] += force[0]*l/(n**2)
            F[2*(i*n)+1] += force[1]*l/(n**2)
    elif dir == 'left':
        for i in range(n):
            F[2*i] += force[0]*l/(n**2)
            F[2*i+1] += force[1]*l/(n**2)
    elif dir == 'right':
        for i in range(n):
            F[2*(i+n*(n-1))] += force[0]*l/(n**2)
            F[2*(i+n*(n-1))+1] += force[1]*l/(n**2)
    return F

def test_bandiwth(K):
    n = K.shape[0]
    bandiwth = n
    nb = 0
    while nb ==0:
        bandiwth -= 1
        nb = 0
        for i in range(n-bandiwth):
            if K[i,i+bandiwth] != 0:
                nb = 1
    
    return bandiwth,bandiwth
            