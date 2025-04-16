""""
This program inted to solve the problem of a plate under load using the finite element method
The program is divided into two parts:
1- the first part is the definition of the functions that will be used in the second part
2- the second part is the implementation of the finite element method to solve the problem of a plate under load
This part is all function used later in the second part
showDeform2D: function to show the deformation of the plate
showMesh2D: function to show the mesh of the plate
animate_plate_oscillation: function to animate the oscillation of the plate
displacement: function to calculate the deformed nodes
L: function to calculate the Lagrange polynomial
Hermit_interpolation: function to calculate the hermite interpolation
shapeFunction: function to calculate the shape function
strainDisplacement: function to calculate the B matrix
local_mass_matrix: function to calculate the local mass matrix
global_mass_matrix: function to calculate the global mass matrix
local_stiffness_matrix: function to calculate the local stiffness matrix
global_stiffness_matrix: function to calculate the global stiffness matrix
numercialIntegration: function to calculate the numerical integration
doubleNumericalIntegration: function to calculate the double numerical integration
doubleNumericalIntergradion: function to calculate the double numerical integration but with the barycentric coordinates
edgeForces: function to apply the forces along the edges of the plate
apply_boundary_conditions: function to apply the boundary conditions
mesh: function to create the mesh
convert_to_banded: function to convert the global stiffness matrix to banded format
calculate_bandwith: function to calculate the bandwith of the matrix to remove the zeros
boundry: function to create the boundary conditions
merge_boudary_conditions: function to merge two boundary conditions
forces: function to apply the forces


Writen by: Arthur Boudehent
Last modification: 2025-01-04
"""

#imports
import matplotlib.pyplot as plt
import matplotlib.animation as an
import numpy as np

def showDeform2D(element,nodes,deformed_nodes):
    x,y,xdef,ydef = [],[],[],[]
    if len(nodes) != len(deformed_nodes):
        raise ValueError("nodesLenghtsIssues")
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

def showMesh2D(element,nodes):
    x,y = [],[]
    for i in element:
        for j in element[i]:
            x.append(nodes[j][0])
            y.append(nodes[j][1])
        x.append(nodes[element[i][0]][0]);x.append(nodes[element[i][2]][0])#close the triangle
        y.append(nodes[element[i][0]][1]);y.append(nodes[element[i][2]][1])
    plt.plot(x, y,marker='x', linestyle='dotted',color='blue')
    plt.axis('equal')
    plt.title("Mesh of a plate")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()

def animate_plate_oscillation(nodes, elements, eigenvectors, frequencies, mode=0, fps=30,scaleTot = 1,timeScale = 1):
    """
    Animate the oscillation of a plate based on a specific mode shape.

    Parameters:
        nodes (np.ndarray): Array of node coordinates.
        elements (dict): Dictionary of elements (triangles).
        eigenvectors (np.ndarray): Array of eigenvectors (mode shapes).
        frequencies (np.ndarray): Array of natural frequencies.
        mode (int): Mode number to visualize (default is 0, the first mode).
        fps (int): Frames per second for the animation.
    """
    # Extract the mode shape and frequency
    mode_shape = eigenvectors[mode]
    omega = frequencies[mode]  # Natural frequency (z)

    # Check if the frequency is valid
    if omega <= 0:
        print(f"Warning: Frequency for mode {mode} is non-positive ({omega}). Skipping animation.")
        return

    # Normalize the mode shape so that the maximum absolute value is 1
    mode_shape_normalized = mode_shape / np.max(np.abs(mode_shape))

    #adjust FPS
    fps = max(fps, int(10 * omega / (2 * np.pi)))  # Ensure at least 10 points per period
    # Time array for the animation
    T = 2 * np.pi / omega  # Period of the mode shape
    t = np.linspace(0, T*timeScale, int(T * fps *timeScale))
    
    # Check if the time array is valid
    if len(t) == 0:
        print(f"Warning: Time array for mode {mode} is empty. Skipping animation.")
        return

    # Scale the mode shape for visualization
    mode_shape_scaled = 0.1*mode_shape_normalized.reshape(-1, 2) * scaleTot  # Apply the scaling factor

    # Create the figure and axis
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title(f"Plate Oscillation - Mode {mode + 1} with a frequency of {omega:.2f} Hz and a scale factor of {scaleTot}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid()

    # Plot the undeformed mesh
    x, y = [], []
    for element in elements.values():
        for node in element:
            x.append(nodes[node][0])
            y.append(nodes[node][1])
        x.append(nodes[element[0]][0]); # Close the triangle
        y.append(nodes[element[0]][1])
        x.append(nodes[element[2]][0])
        y.append(nodes[element[2]][1])
    undeformed_plot, = ax.plot(x, y, linestyle='dotted', color='blue', label='Original')

    # Plot the deformed mesh
    deformed_plot, = ax.plot([], [], linestyle='dotted', color='red', label='Deformed')

    ax.legend()

    # Update function for the animation
    def update(frame):
        time = t[frame]
        displacement = np.sin(omega * time) * mode_shape_scaled
        deformed_nodes = nodes + displacement

        # Update the deformed mesh
        x_def, y_def = [], []
        for element in elements.values():
            for node in element:
                x_def.append(deformed_nodes[node][0])
                y_def.append(deformed_nodes[node][1])
            x_def.append(deformed_nodes[element[0]][0])  # Close the triangle
            y_def.append(deformed_nodes[element[0]][1])
            x_def.append(deformed_nodes[element[2]][0])
            y_def.append(deformed_nodes[element[2]][1])
        deformed_plot.set_data(x_def, y_def)
        return deformed_plot,
    
    # Create the animation
    anim = an.FuncAnimation(fig, update, frames=len(t), interval=1000 / fps, blit=True)

    # Show the animation
    plt.show()

def displacement(nodes,U,scale = 1):  
    deformed_nodes = np.copy(nodes)
    for i in range(len(nodes)):
        deformed_nodes[i] = [nodes[i][0]+U[2*i]*scale,nodes[i][1]+U[2*i+1]*scale]#adding the displacement to the original coordinates
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

def numercialIntegration(f:callable, a, b, n):#numerical integration
    if f is None or not callable(f):
        raise ValueError("f must be a callable function")
    h = (b - a) / n
    result = 0
    for i in range(n):
        result += f(a + i * h) * h
    return result

def doubleNumericalIntegration(f: callable, n: int, points: list):#double numerical integration with the barycentric coordinates
    """
    Perform double numerical integration over a triangular surface defined by 3 points.

    Parameters:
        f (callable): Function to integrate. It must take two arguments (x, y).
        n (int): Number of integration points per dimension (controls accuracy).
        points (list): List of 3 points defining the triangle, e.g., [(x1, y1), (x2, y2), (x3, y3)].

    Returns:
        float: Approximation of the integral over the triangular surface.
    """
    if len(points) != 3:
        raise ValueError("The 'points' parameter must contain exactly 3 points defining the triangle.")

    # Extract the points
    (x1, y1), (x2, y2), (x3, y3) = points

    # Calculate the area of the triangle
    area = 0.5 * abs(
        x1 * (y2 - y3) +
        x2 * (y3 - y1) +
        x3 * (y1 - y2)
    )

    # Generate integration points using barycentric coordinates
    result = 0
    for i in range(n):
        for j in range(n - i):
            # Barycentric coordinates (l1, l2, l3)
            l1 = i / n
            l2 = j / n
            l3 = 1 - l1 - l2

            # Map barycentric coordinates to (x, y)
            x = l1 * x1 + l2 * x2 + l3 * x3
            y = l1 * y1 + l2 * y2 + l3 * y3

            # Evaluate the function at (x, y)
            result += f(x, y)

    # Scale by the area of the triangle and the integration weight
    result *= area / (n * n)

    return result

def doubleNumericalIntergradion(f:callable,n,m,c,d,a=1,b=1,g_lower:callable = lambda x:1,g_upper:callable = lambda x:1):#double numerical integration
    if not callable(f):
        raise ValueError("f must be a callable function")
    if not callable(g_upper):
        raise ValueError("g_upper must be a callable function")
    if not callable(g_lower):
        raise ValueError("g_lower must be a callable function")
    if a >= b or c >= d:
        raise ValueError("a must be less than b and c must be less than d")
    if c==d:
        return 0
    
    
    hy = (d - c) / m
    result = 0
    
    for j in range(m):
        y = c + j * hy
        upper_bound_x = g_upper(y)*b
        lower_boud_x = g_lower(y)*a
        hx = (upper_bound_x - lower_boud_x) / n
        for i in range(n):
            x = a + i * hx
            result += f(x, y) * hx * hy
    return result

def shapeFunctionCoeff(nodes, elements,nbElement):
    numpoints = elements[nbElement]
    L=[]
    delta = (nodes[numpoints[0]][0]*(nodes[numpoints[1]][1]-nodes[numpoints[2]][1])
                +nodes[numpoints[1]][0]*(nodes[numpoints[2]][1]-nodes[numpoints[0]][1])
                +nodes[numpoints[2]][0]*(nodes[numpoints[0]][1]-nodes[numpoints[1]][1]))
    for i in range(len(numpoints)):# calulation of each coefficient of the shape function
        a = nodes[numpoints[(i+1)%3]][0]*nodes[numpoints[(i+2)%3]][1]-nodes[numpoints[(i+2)%3]][0]*nodes[numpoints[(i+1)%3]][1]
        b = nodes[numpoints[(i+1)%3]][1]-nodes[numpoints[(i+2)%3]][1]
        c = nodes[numpoints[(i+2)%3]][0]-nodes[numpoints[(i+1)%3]][0]
        L.append([a/delta,b/delta,c/delta])
    return L

def shapeFunction(nodes,elements,nbElement):#shape function
    L = shapeFunctionCoeff(nodes,elements,nbElement)
    def N(x,y):
        N = []
        for i in range(len(L)):
            N.append(L[i][0]+L[i][1]*x+L[i][2]*y)
        return N
    return N

def strainDisplacement(nodes,elements,nbElement): #calcul of the B matrix, since the shape function is linear, the B matrix is constant
    points = elements[nbElement]
    L = shapeFunctionCoeff(nodes,elements,nbElement)
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
    D = np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])*E/(1-nu**2)# to remove from the function because it depends on the type of material
    Ke = np.dot(B.T,np.dot(D,B))*Volume #since the B matrix is constant, the integral of B^TDB is equal to B^TDB*Volume
    return Ke

def global_stiffness_matrix(nodes,element,E,nu,t):
    n = len(nodes)
    K = np.zeros((2*n,2*n),dtype = float)
    for i in element:
        Ke = local_stiffness_matrix(nodes,element,i,E,nu,t)
        for j in range(3):
            for k in range(3):
                K[2*element[i][j],2*element[i][k]] += Ke[2*j,2*k]
                K[2*element[i][j],2*element[i][k]+1] += Ke[2*j,2*k+1]
                K[2*element[i][j]+1,2*element[i][k]] += Ke[2*j+1,2*k]
                K[2*element[i][j]+1,2*element[i][k]+1] += Ke[2*j+1,2*k+1]
    return K

def reduce_mass_matrix(M, constrained_dofs):
    """
    Reduce the mass matrix by removing rows and columns corresponding to constrained DOFs.

    Parameters:
        M (np.ndarray): Full mass matrix.
        constrained_dofs (list): List of constrained degrees of freedom (DOFs).

    Returns:
        M_reduced (np.ndarray): Reduced mass matrix.
        free_dofs (list): List of free degrees of freedom (DOFs).
    """
    # Determine free DOFs
    total_dofs = M.shape[0]
    free_dofs = [i for i in range(total_dofs) if i not in constrained_dofs]

    # Reduce the mass matrix
    M_reduced = M[np.ix_(free_dofs, free_dofs)]

    return M_reduced

def reconstruct_full_vector(reduced_vector, constrained_dofs, total_dofs):
    
    full_vector = np.zeros(total_dofs)
    free_dofs = [i for i in range(total_dofs) if i not in constrained_dofs]

    for i, free_i in enumerate(free_dofs):
        full_vector[free_i] = reduced_vector[i]

    return full_vector

def reconstruct_full_vectors(reduced_vectors, constrained_dofs, total_dofs):
    """
    Reconstruct the full eigenvectors, preserving the mode shapes.
    """
    num_modes = reduced_vectors.shape[1]
    full_vectors = np.zeros((num_modes, total_dofs))  # Note the shape change
    free_dofs = [i for i in range(total_dofs) if i not in constrained_dofs]
    
    for i in range(num_modes):
        mode = np.zeros(total_dofs)
        mode[free_dofs] = reduced_vectors[:, i]
        full_vectors[i] = mode
        
    return full_vectors

def reconstruct_full_matrix(reduced_matrix, constrained_dofs, total_dofs):
    """
    Reconstruct the full matrix (e.g., stiffness or mass matrix) from its reduced form.

    Parameters:
        reduced_matrix (np.ndarray): Reduced matrix.
        constrained_dofs (list): List of constrained degrees of freedom (DOFs).
        total_dofs (int): Total number of DOFs in the original system.

    Returns:
        full_matrix (np.ndarray): Full matrix.
    """
    full_matrix = np.zeros((total_dofs, total_dofs))
    free_dofs = [i for i in range(total_dofs) if i not in constrained_dofs]

    for i, free_i in enumerate(free_dofs):
        for j, free_j in enumerate(free_dofs):
            full_matrix[free_i, free_j] = reduced_matrix[i, j]

    return full_matrix

def mesh(L,l,N,M=0,offsetX=0,offsetY=0):
    if(M==0):
        M=N
    n = N+1
    m= M+1
    nodes = np.zeros((n*m,2))
    elements = {}
    for i in range(n):
        for j in range(m):
            nodes[i*m+j] = [i*L/(n-1)-offsetX,j*l/(m-1)-offsetY]
    index = 0
    for i in range(n-1):
        for j in range(m-1):
            elements[index] = [j + i * m, j + i * m + m, j + i * m + 1]
            index += 1
            elements[index] = [j+i*m+m+1,j+i*m+1,j+i*m+m]  
            index+=1        
    return nodes,elements

def convert_to_banded(K, lower_bandwidth, upper_bandwidth):
    n = K.shape[0]
    ab = np.zeros((lower_bandwidth + upper_bandwidth + 1, n))
    for i in range(n):
        for j in range(max(0, i - lower_bandwidth), min(n, i + upper_bandwidth + 1)):
            ab[upper_bandwidth + i - j, j] = K[i, j]
    return ab

def convert_to_banded_optimized(K,lower_bandwidth, upper_bandwidth):
    n = K.shape[0]
    ab = np.zeros((lower_bandwidth + upper_bandwidth + 1, n))
    for i in range(n):
        start = max(0, i - lower_bandwidth)
        end = min(n, i + upper_bandwidth + 1)
        ab[upper_bandwidth + i - np.arange(start, end), np.arange(start, end)] = K[i, start:end]
    return ab

def calculate_bandwidth(K):
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

def apply_boundary_conditions(K, F, bc):
   
    # Flatten the list of constrained degrees of freedom (DOFs)
    constrained_dofs = []
    for node, displacement in bc:
        if displacement[0] is not None:# Skip the node if the displacement is negative      
            constrained_dofs.append(2 * node)       # x DOF
        if displacement[1] is not None:# Skip the node if the displacement is negative
            constrained_dofs.append(2 * node + 1)  # y DOF
    # Create a mask for the DOFs to keep
    total_dofs = K.shape[0]
    free_dofs = [i for i in range(total_dofs) if i not in constrained_dofs]

    # Reduce the stiffness matrix and force vector
    K_reduced = K[np.ix_(free_dofs, free_dofs)]
    F_reduced = F[free_dofs]

    return K_reduced, F_reduced, constrained_dofs

def boundry(direct,disp,N,M=0):
    if M == 0:
        M=N
    n = N+1
    m= M+1
    boundry = []
    if direct == 'top':
        for i in range(n):
            boundry.append((i*m+m,disp))
    elif direct == 'bottom':
        for i in range(n):
            boundry.append((i*m,disp))
    elif direct == 'left':
        for i in range(m):
            boundry.append((i,disp))
    elif direct == 'right':
        for i in range(m):
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

def edgeForces(F, force, dir, l,N,M=0):# function to apply the forces along the edges of the plate
    if M == 0:
        M = N
    n = N + 1
    m = M + 1
    if dir == 'top':
        for i in range(n):
            F[2 * (i * m + m - 1)] += force[0] * l / n
            F[2 * (i * m + m - 1) + 1] += force[1] * l / n
    elif dir == 'bottom':
        for i in range(n):
            F[2 * (i * m)] += force[0] * l / n
            F[2 * (i * m) + 1] += force[1] * l / n
    elif dir == 'left':
        for i in range(m):
            F[2 * i] += force[0] * l / m
            F[2 * i + 1] += force[1] * l / m
    elif dir == 'right':
        for i in range(m):
            node_index = i + n * (m - 1)  # Global node index for the 'right' edge
            F[2 * node_index] += force[0] * l / m
            F[2 * node_index + 1] += force[1] * l / m
    return F

def globalForces(F,force,nodes,elements,t):
    for i in elements:
        nbPoints = elements[i]
        points = [nodes[nbPoints[j]] for j in range(3)]
        Fx += doubleNumericalIntegration(lambda x,y:force[0]*shapeFunction(nodes,elements,i)(x,y),10,points)
        Fy += doubleNumericalIntegration(lambda x,y:force[1]*shapeFunction(nodes,elements,i)(x,y),10,points)
        for j in nbPoints:    
            F[2*nbPoints[0]] += Fx*t/3
            F[2*nbPoints[0]+1] += Fy*t/3
    return F

def globalEnergy(U,K):
    return 0.5*np.dot(U.T,np.dot(K,U))

def calculate_bandwidth_optimized(K):
    n = K.shape[0]
    bandiwth = n
    nb = 0
    while nb ==0:
        bandiwth -= 1
        nb = 0
        for i in range(n-bandiwth):
            if K[i,i+bandiwth] != 0:
                nb = 1
                break
    return bandiwth,bandiwth

Me = np.zeros((6,6))
for i in range(3):
    for j in range(3):
        if i == j:
            Me[2*i,2*j] = 2
            Me[2*i+1,2*j+1] = 2
        else:
            Me[2*i,2*j] = 1
            Me[2*i+1,2*j+1] = 1
            
def localMassMtrix(nodes,elements,nbElement,rho,t):
    points = elements[nbElement]
    Volume = 0.5*(nodes[points[0]][0]*(nodes[points[1]][1]-nodes[points[2]][1])
                +nodes[points[1]][0]*(nodes[points[2]][1]-nodes[points[0]][1])
                +nodes[points[2]][0]*(nodes[points[0]][1]-nodes[points[1]][1]))*t#calcul of the triangle volume
    Mk = np.zeros((6,6))
    Mk = Me*rho*Volume/12
    return Mk

def global_mass_matrix(nodes,element,rho,t):
    n = len(nodes)
    M = np.zeros((2*n,2*n))
    for i in element:
        Mk = localMassMtrix(nodes,element,i,rho,t)
        for j in range(3):
            for k in range(3):
                M[2*element[i][j],2*element[i][k]] += Mk[2*j,2*k]
                M[2*element[i][j],2*element[i][k]+1] += Mk[2*j,2*k+1]
                M[2*element[i][j]+1,2*element[i][k]] += Mk[2*j+1,2*k]
                M[2*element[i][j]+1,2*element[i][k]+1] += Mk[2*j+1,2*k+1]
    return M

def normalize_matrix(matrix):
    max_value = np.max(np.abs(matrix))
    if max_value == 0:
        return matrix,max_value
    return matrix / max_value,max_value