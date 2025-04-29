"""
Finite Element Method (FEM) Implementation for 2D Structural Analysis

This module implements a comprehensive FEM solver for 2D structural analysis,
supporting both triangular and quadrilateral elements with linear and quadratic shape functions.

Key Features:
-------------
1. Mesh Generation
    - Triangular and quadrilateral elements
    - Coarse and fine mesh options
    - Support for structured 2D grids

2. Analysis Capabilities
    - Static analysis
    - Modal analysis
    - Natural frequency calculation
    - Mode shape visualization
    - Plane stress problems

3. Visualization Tools
    - Mesh visualization
    - Deformation plotting
    - Mode shape animation
    - Interactive plots

Core Functions:
--------------
Mesh Operations:
    - mesh(): Generate 2D structured mesh
    - showMesh2D(): Visualize mesh
    - listPlots(): Generate plotting coordinates

Analysis:
    - global_stiffness_matrix(): Assemble global stiffness matrix
    - global_mass_matrix(): Assemble global mass matrix
    - apply_boundary_conditions(): Apply constraints
    - shapeFunction(): Calculate shape functions
    - strainDisplacement(): Calculate B matrix

Visualization:
    - showDeform2D(): Plot deformed structure
    - animate_plate_oscillation(): Animate mode shapes

Utilities:
    - doubleNumericalIntegration(): Numerical integration
    - shapeFunctionCoeff(): Shape function coefficients
    - edgeForces(): Apply edge loads

Author: Arthur Boudehent
Institution: La Sapienza University, Rome
Created as part of internship project
Last modified: 2025-18-04
"""

#imports
import matplotlib.pyplot as plt
import matplotlib.animation as an
import numpy as np
import sympy as sp

def mesh(L,l,N,M=0,offsetX=0,offsetY=0,element_type='tri',mesh_type = 'coarse'):

    if(M==0):
        M=N
    if(mesh_type == 'coarse'):
        n = N+1
        m= M+1
    if(mesh_type == 'fine'):
        n = N*2+1
        m= M*2+1
    
    nodes = np.zeros((n*m,2))
    elements = {}
    for i in range(n):
        for j in range(m):
            nodes[i*m+j] = [i*L/(n-1)-offsetX,j*l/(m-1)-offsetY]
    if(element_type == 'tri'):
        if(mesh_type=='coarse'):
            index = 0
            for i in range(N):
                for j in range(M):
                    elements[index] = [j + i * m, j + i * m + m, j + i * m + 1]
                    index += 1
                    elements[index] = [j+i*m+m+1,j+i*m+1,j+i*m+m]  
                    index+=1
        if(mesh_type=='fine'):
            index = 0
            for i in range(N):
                for j in range(M):
                    elements[index] = [2*j + 2*i * m, 2*j + 2*i * m + m, 2*j + 2*i * m + 2*m, 2*j + 2*i * m + m + 1,2*j+2*i*m+2,2*j+2*i*m+1]
                    index += 1
                    elements[index] = [2*j + (2*i+2)* m + 2, 2*j + (2*i+1)* m + 2, 2*j + (2*i) * m + 2, 2*j + (2*i+1)*m +1,2*j+(2*i+2)*m,2*j+(2*i+2)*m+1]  
                    index+=1     
    if(element_type=='quad'):
        if(mesh_type=='coarse'):
            index = 0
            for i in range(N):
                for j in range(M):
                    elements[index] = [j + i * m, j + i * m + m, j + i * m + m+1,j + i * m +1]
                    index += 1
        if(mesh_type=='fine'):
            index = 0
            for i in range(N):
                for j in range(M):
                    elements[index] = [2*j + 2*i * m,2*j + (2*i+1) * m,2*j+(2*i+2) * m,2*j+(2*i+2) * m+1,2*j +(2*i+2) * m+2,2*j+(2*i+1) * m+2,2*j + (2*i) * m + 2,j*2+(2*i) * m + 1,2*j + (2*i+1) * m + 1]
                    index += 1   
    return nodes,elements

def listPlots(nodes,elements,element_type = 'tri',mesh_type = 'coarse'):
    x,y = [],[]
    interior_x, interior_y = [], [] 
    if element_type == 'tri':

        #add all points of the mesh
        for i in elements:
            for j in elements[i]:
                x.append(nodes[j][0])
                y.append(nodes[j][1])

            #close the triangle
            if mesh_type == 'coarse':
                x.append(nodes[elements[i][0]][0]);y.append(nodes[elements[i][0]][1])#add local node nb 0
                x.append(nodes[elements[i][2]][0]);y.append(nodes[elements[i][2]][1])#add local node nb 2
            if mesh_type == 'fine':
                if i%2 == 0:
                    x.append(nodes[elements[i][0]][0]);y.append(nodes[elements[i][0]][1])#add local node nb 0
                    x.append(nodes[elements[i][2]][0]);y.append(nodes[elements[i][2]][1])#add local node nb 2 
                else:
                    x.append(nodes[elements[i][5]][0]);y.append(nodes[elements[i][5]][1])#add local node nb 5
                    x.append(nodes[elements[i][0]][0]);y.append(nodes[elements[i][0]][1])#add local node nb 0
    if element_type == 'quad' and mesh_type == 'coarse':
        #add all points of the mesh
        for i in elements:
            for j in elements[i]:
                x.append(nodes[j][0])
                y.append(nodes[j][1])
            x.append(nodes[elements[i][0]][0]);y.append(nodes[elements[i][0]][1]) #add local node nb 0, the bottom right node
            x.append(nodes[elements[i][1]][0]);y.append(nodes[elements[i][1]][1]) #add local node nb 1, the bottom left node
            x.append(nodes[elements[i][2]][0]);y.append(nodes[elements[i][2]][1]) #add local node nb 2, the top left node
    if element_type == 'quad' and mesh_type == 'fine':
        #add all points of the mesh exept the last one 
        for i in elements:
            for j in range(len(elements[i])-1):
                k = elements[i][j]
                x.append(nodes[k][0])
                y.append(nodes[k][1])
            x.append(nodes[elements[i][0]][0]);y.append(nodes[elements[i][0]][1]) #add local node nb 0, the bottom right node
            x.append(nodes[elements[i][2]][0]);y.append(nodes[elements[i][2]][1]) #add local node nb 1, the bottom left node
            x.append(nodes[elements[i][4]][0]);y.append(nodes[elements[i][4]][1]) #add local node nb 4, the top left node
            interior_node = elements[i][-1]  # Get the last node
            interior_x.append(nodes[interior_node][0]);interior_y.append(nodes[interior_node][1])
    
    return x,y,interior_x,interior_y

def showDeform2D(nodes,deformed_nodes,element,element_type = 'tri',mesh_type = 'coarse',show = True):
    x,y,xdef,ydef = [],[],[],[]
    intx,inty,intX,intY = [],[],[],[]
    if len(nodes) != len(deformed_nodes):
        raise ValueError("nodesLenghtsIssues")
    x,y,intx,inty = listPlots(nodes,element,element_type,mesh_type)
    xdef,ydef,intX,intY = listPlots(deformed_nodes,element,element_type,mesh_type)

    if intx:  # Only plot if there are interior points
        plt.plot(intx, inty, 'x', linestyle='',color = 'blue')
        plt.plot(intX, intY, 'x', linestyle='',color = 'red')

    plt.plot(x, y,marker='x', linestyle='dotted',color='blue',label='original')
    plt.plot(xdef, ydef,marker='x', linestyle='dotted',color='red',label='deformed')
    plt.axis('equal')
    plt.title("deformation of a plate under load")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    if show:
        plt.show()

def showMesh2D(nodes,element,element_type = 'tri',mesh_type = 'coarse',show = True):
    x, y, interior_x, interior_y = listPlots(nodes, element, element_type, mesh_type)
    
    plt.plot(x, y, marker='x', linestyle='dotted', color='blue')
    if interior_x:  # Only plot if there are interior points
        plt.plot(interior_x, interior_y, 'x', linestyle='',color = 'blue')
    
    plt.plot(x, y,marker='x', linestyle='dotted',color='blue')
    plt.axis('equal')
    plt.title("Mesh of a plate")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    if show:
        plt.show()

def animate_plate_oscillation(nodes, elements, eigenvectors, frequencies, mode=0, fps=30,scaleTot = 1,timeScale = 1, element_type='tri', mesh_type='coarse', save_animation=False, filename=None):
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
    # Plot undeformed mesh
    x, y, interior_x, interior_y = listPlots(nodes, elements, element_type, mesh_type)
    undeformed_plot, = ax.plot(x, y, linestyle='dotted', marker='x', color='blue', label='Original')
    if interior_x:  # Plot interior nodes if they exist
        interior_plot = ax.plot(interior_x, interior_y, marker='x', color='blue', linestyle='')

    # Initialize deformed plot
    deformed_plot, = ax.plot([], [], linestyle='dotted', marker='x', color='red', label='Deformed')
    if interior_x:  # Initialize deformed interior nodes plot
        deformed_interior_plot, = ax.plot([], [], marker='x', color='red', linestyle='')

   # ax.legend()

    def update(frame):
        time = t[frame]
        displacement = np.sin(omega * time) * mode_shape_scaled
        deformed_nodes = nodes + displacement

        # Update deformed mesh
        x_def, y_def, int_x_def, int_y_def = listPlots(deformed_nodes, elements, element_type, mesh_type)
        deformed_plot.set_data(x_def, y_def)

        if interior_x:  # Update interior nodes if they exist
            deformed_interior_plot.set_data(int_x_def, int_y_def)
            return deformed_plot, deformed_interior_plot
        return (deformed_plot,)

    # Create animation
    anim = an.FuncAnimation(fig, update, frames=len(t), interval=1000/fps, blit=True)
    
    if save_animation:
        if filename is None:
            filename = f'mode_{mode+1}.gif'
        print(f"Saving animation to {filename}...")
        anim.save(filename, writer='pillow', fps=fps)
        plt.close()
    else:
        plt.show()

def displacement(nodes,U,scale = 1):  
    deformed_nodes = np.copy(nodes)
    for i in range(len(nodes)):
        deformed_nodes[i] = [nodes[i][0]+U[2*i]*scale,nodes[i][1]+U[2*i+1]*scale]#adding the displacement to the original coordinates
    return deformed_nodes

def shape_functions(element_type='tri', mesh_type='coarse'):
    """
    Generate shape functions for different element types using sympy.
    
    Parameters:
        element_type: 'tri' or 'quad'
        mesh_type: 'coarse' or 'fine'
    
    Returns:
        N: List of shape functions
        xi, eta: Symbolic coordinates
    """
    
    # Define symbolic variables
    xi, eta = sp.symbols('xi eta')
    
    if element_type == 'tri':
        if mesh_type == 'coarse':
            # Linear triangle (3 nodes)
            N = [
                1 - xi - eta,  # N1
                xi,           # N2
                eta          # N3
            ]
        else:
            # Quadratic triangle (6 nodes)
            N = [
                (1 - xi - eta) * (2*(1 - xi - eta) - 1),  # N1
                xi * (2*xi - 1),                          # N2
                eta * (2*eta - 1),                        # N3
                4 * xi * (1 - xi - eta),                  # N4
                4 * xi * eta,                             # N5
                4 * eta * (1 - xi - eta)                  # N6
            ]
    
    elif element_type == 'quad':
        if mesh_type == 'coarse':
            # Linear quad (4 nodes)
            N = [
                (1 - xi) * (1 - eta) / 4,  # N1
                (1 + xi) * (1 - eta) / 4,  # N2
                (1 + xi) * (1 + eta) / 4,  # N3
                (1 - xi) * (1 + eta) / 4   # N4
            ]
        else:
            # Quadratic quad (9 nodes)
            N = [
                xi * (xi - 1) * eta * (eta - 1) / 4,  # Corner nodes
                xi * (xi + 1) * eta * (eta - 1) / 4,
                xi * (xi + 1) * eta * (eta + 1) / 4,
                xi * (xi - 1) * eta * (eta + 1) / 4,
                (1 - xi*xi) * eta * (eta - 1) / 2,    # Midside nodes
                xi * (xi + 1) * (1 - eta*eta) / 2,
                (1 - xi*xi) * eta * (eta + 1) / 2,
                xi * (xi - 1) * (1 - eta*eta) / 2,
                (1 - xi*xi) * (1 - eta*eta)           # Center node
            ]
    
    # Simplify expressions
    N = [sp.simplify(n) for n in N]
    
    return N, xi, eta

def get_shape_derivatives(element_type='tri', mesh_type='coarse'):
    """
    Calculate derivatives of shape functions with respect to xi and eta.
    """
    N, xi, eta = shape_functions(element_type, mesh_type)
    
    # Calculate derivatives
    dN_dxi = [sp.diff(n, xi) for n in N]
    dN_deta = [sp.diff(n, eta) for n in N]
    
    # Simplify expressions
    dN_dxi = [sp.simplify(d) for d in dN_dxi]
    dN_deta = [sp.simplify(d) for d in dN_deta]
    
    return dN_dxi, dN_deta

def create_shape_functions(element_type='tri', mesh_type='coarse'):
    """
    Create numerical functions from symbolic shape functions.
    """
    N, xi, eta = shape_functions(element_type, mesh_type)
    dN_dxi, dN_deta = get_shape_derivatives(element_type, mesh_type)
    
    # Convert to numerical functions
    N_funcs = [sp.lambdify((xi, eta), n, 'numpy') for n in N]
    dN_dxi_funcs = [sp.lambdify((xi, eta), d, 'numpy') for d in dN_dxi]
    dN_deta_funcs = [sp.lambdify((xi, eta), d, 'numpy') for d in dN_deta]
    
    return N_funcs, dN_dxi_funcs, dN_deta_funcs

def get_gauss_points(element_type='tri'):
    """Get Gaussian quadrature points and weights for different element types"""
    if element_type == 'tri':
        # Use 7-point Gauss quadrature for triangles
        points = [
            (1/3, 1/3),                          # Point 1
            (0.797426985353087, 0.101286507323456),  # Point 2
            (0.101286507323456, 0.797426985353087),  # Point 3
            (0.101286507323456, 0.101286507323456),  # Point 4
            (0.470142064105115, 0.470142064105115),  # Point 5
            (0.470142064105115, 0.059715871789770),  # Point 6
            (0.059715871789770, 0.470142064105115)   # Point 7
        ]
        weights = [
            0.225,                     # Weight 1
            0.125939180544827,        # Weight 2
            0.125939180544827,        # Weight 3
            0.125939180544827,        # Weight 4
            0.132394152788506,        # Weight 5
            0.132394152788506,        # Weight 6
            0.132394152788506         # Weight 7
        ]
    else:  # quad
        # 3x3 Gauss points for quadrilateral (more accurate)
        a = np.sqrt(0.6)
        points = [
            (-a, -a), (0, -a), (a, -a),
            (-a,  0), (0,  0), (a,  0),
            (-a,  a), (0,  a), (a,  a)
        ]
        w1 = 5/9
        w2 = 8/9
        w3 = 5/9
        weights = [
            w1*w1, w1*w2, w1*w3,
            w2*w1, w2*w2, w2*w3,
            w3*w1, w3*w2, w3*w3
        ]
    
    return points, weights

def element_stiffness_matrix(nodes, elements, element_number, E, nu, t, element_type='tri', mesh_type='coarse', integration='gauss'):
    """
    Compute element stiffness matrix using either Gaussian or symbolic integration.
    
    Parameters:
        nodes, elements, element_number: Mesh data
        E, nu, t: Material and geometric properties
        element_type: 'tri' or 'quad'
        mesh_type: 'coarse' or 'fine'
        integration: 'gauss' or 'symbolic'
    """
    # For linear triangles, use direct computation
    if element_type == 'tri' and mesh_type == 'coarse':
        return local_stiffness_matrix(nodes, elements, element_number, E, nu, t)
    
    # Material matrix D for plane stress
    D = np.array([[1, nu, 0],
                  [nu, 1, 0],
                  [0, 0, (1-nu)/2]]) * (E/(1-nu**2))
    
    element_nodes = elements[element_number]
    n_nodes = len(element_nodes)
    Ke = np.zeros((2*n_nodes, 2*n_nodes))
    
    if integration == 'gauss':
        # Use Gaussian integration
        gauss_points, weights = get_gauss_points(element_type)
        for point, weight in zip(gauss_points, weights):
            B, J, detJ = compute_B_matrix_at_point(nodes, elements, element_number, 
                                                 point[0], point[1], 
                                                 element_type, mesh_type)
            Ke += weight * B.T @ D @ B * abs(detJ) * t
    
    elif integration == 'symbolic':
        # Use symbolic integration
        xi, eta = sp.symbols('xi eta')
        
        # Get shape functions and derivatives symbolically
        dN_dxi, dN_deta = get_shape_derivatives(element_type, mesh_type)
        
        # Calculate Jacobian symbolically
        node_coords = np.array([nodes[i] for i in element_nodes])
        J = sp.zeros(2, 2)
        for i in range(n_nodes):
            J[0,0] += dN_dxi[i] * node_coords[i,0]
            J[0,1] += dN_dxi[i] * node_coords[i,1]
            J[1,0] += dN_deta[i] * node_coords[i,0]
            J[1,1] += dN_deta[i] * node_coords[i,1]
        
        detJ = sp.simplify(J.det())
        Jinv = J.inv()
        
        # Construct B matrix symbolically
        B = sp.zeros(3, 2*n_nodes)
        for i in range(n_nodes):
            dNi_dx = sp.simplify(Jinv[0,0] * dN_dxi[i] + Jinv[0,1] * dN_deta[i])
            dNi_dy = sp.simplify(Jinv[1,0] * dN_dxi[i] + Jinv[1,1] * dN_deta[i])
            
            B[0, 2*i] = dNi_dx
            B[0, 2*i+1] = 0
            B[1, 2*i] = 0
            B[1, 2*i+1] = dNi_dy
            B[2, 2*i] = dNi_dy
            B[2, 2*i+1] = dNi_dx
        
        # Compute integrand symbolically
        integrand = B.T * D * B * sp.Abs(detJ) * t
        
        # Perform symbolic integration
        if element_type == 'tri':
            # Integrate over unit triangle
            for i in range(2*n_nodes):
                for j in range(2*n_nodes):
                    integral = sp.integrate(
                        sp.integrate(integrand[i,j], (eta, 0, 1-xi)),
                        (xi, 0, 1)
                    )
                    Ke[i,j] = float(sp.simplify(integral))
        else:
            # Integrate over unit square
            for i in range(2*n_nodes):
                for j in range(2*n_nodes):
                    integral = sp.integrate(
                        sp.integrate(integrand[i,j], (eta, -1, 1)),
                        (xi, -1, 1)
                    )
                    Ke[i,j] = float(sp.simplify(integral))
    
    else:
        raise ValueError("Integration method must be either 'gauss' or 'symbolic'")
    
    return Ke

def compute_B_matrix_at_point(nodes, elements, element_number, xi, eta, element_type='tri', mesh_type='coarse'):
    """
    Compute B matrix at a specific point (xi, eta).
    Uses the same logic as compute_B_matrix but evaluates at given point.
    """
    # Get shape function derivatives
    dN_dxi, dN_deta = get_shape_derivatives(element_type, mesh_type)
    
    # Get element nodes
    element_nodes = elements[element_number]
    node_coords = np.array([nodes[i] for i in element_nodes])
    
    # Initialize matrices
    n_nodes = len(element_nodes)
    B = np.zeros((3, 2*n_nodes))  # 3 strain components, 2 DOFs per node
    
    # Calculate Jacobian matrix
    J = np.zeros((2, 2))
    for i in range(n_nodes):
        dNi_dxi = float(dN_dxi[i].subs({'xi': xi, 'eta': eta}))
        dNi_deta = float(dN_deta[i].subs({'xi': xi, 'eta': eta}))
        
        J[0,0] += dNi_dxi * node_coords[i,0]   # dx/dxi
        J[0,1] += dNi_dxi * node_coords[i,1]   # dy/dxi
        J[1,0] += dNi_deta * node_coords[i,0]  # dx/deta
        J[1,1] += dNi_deta * node_coords[i,1]  # dy/deta
    
    # Calculate determinant and inverse of Jacobian
    detJ = np.linalg.det(J)
    if abs(detJ) < 1e-10:
        raise ValueError(f"Jacobian determinant near zero for element {element_number}")
    
    Jinv = np.linalg.inv(J)
    
    # Compute B matrix components for each node
    for i in range(n_nodes):
        dNi_dxi = float(dN_dxi[i].subs({'xi': xi, 'eta': eta}))
        dNi_deta = float(dN_deta[i].subs({'xi': xi, 'eta': eta}))
        
        # Transform derivatives to global coordinates
        dNi_dx = Jinv[0,0] * dNi_dxi + Jinv[0,1] * dNi_deta
        dNi_dy = Jinv[1,0] * dNi_dxi + Jinv[1,1] * dNi_deta
        
        # Fill B matrix
        B[0, 2*i] = dNi_dx
        B[0, 2*i+1] = 0
        B[1, 2*i] = 0
        B[1, 2*i+1] = dNi_dy
        B[2, 2*i] = dNi_dy
        B[2, 2*i+1] = dNi_dx
    
    return B, J, detJ

def global_stiffness_matrix(nodes, elements, E, nu, t,element_type = 'tri',mesh_type = 'coarse',integration = 'gauss'):
    """
    Assemble the global stiffness matrix for the entire mesh.
    """
    n_nodes = len(nodes)
    K_global = np.zeros((2*n_nodes, 2*n_nodes))
    
    for element_number in elements:
        Ke = element_stiffness_matrix(nodes, elements, element_number, E, nu, t,element_type,mesh_type,integration)
        element_nodes = elements[element_number]
        
        for i in range(len(element_nodes)):
            for j in range(len(element_nodes)):
                K_global[2*element_nodes[i], 2*element_nodes[j]] += Ke[2*i, 2*j]
                K_global[2*element_nodes[i]+1, 2*element_nodes[j]] += Ke[2*i+1, 2*j]
                K_global[2*element_nodes[i], 2*element_nodes[j]+1] += Ke[2*i, 2*j+1]
                K_global[2*element_nodes[i]+1, 2*element_nodes[j]+1] += Ke[2*i+1, 2*j+1]
    
    return K_global

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

def global_stiffness_matrix_linear(nodes,element,E,nu,t):
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

def boundary(direct, disp, N, M=0, element_type='tri', mesh_type='coarse'):
    """
    Generate boundary conditions for edges of the mesh.
    
    Parameters:
        direct (str): Direction ('top', 'bottom', 'left', 'right')
        disp (tuple): Displacement vector (dx, dy)
        N (int): Number of elements in x direction
        M (int): Number of elements in y direction
        element_type (str): Element type ('tri' or 'quad')
        mesh_type (str): Mesh type ('coarse' or 'fine')
    
    Returns:
        list: List of tuples (node_index, displacement)
    """
    if M == 0:
        M = N
    
    # Calculate number of nodes per side
    if mesh_type == 'coarse':
        n = N + 1  # nodes in x direction
        m = M + 1  # nodes in y direction
    else:  # fine mesh
        n = 2*N + 1
        m = 2*M + 1
    
    boundary = []
    
    if direct == 'top':
        if mesh_type == 'coarse':
            for i in range(n):
                boundary.append((i*m + m-1, disp))
        else:
            for i in range(2*N + 1):
                boundary.append((i*m + m-1, disp))
                
    elif direct == 'bottom':
        if mesh_type == 'coarse':
            for i in range(n):
                boundary.append((i*m, disp))
        else:
            for i in range(2*N + 1):
                boundary.append((i*m, disp))
                
    elif direct == 'left':
        if mesh_type == 'coarse':
            for i in range(m):
                boundary.append((i, disp))
        else:
            for i in range(2*M + 1):
                boundary.append((i, disp))
                
    elif direct == 'right':
        if mesh_type == 'coarse':
            for i in range(m):
                boundary.append((i + (n-1)*m, disp))
        else:
            for i in range(2*M + 1):
                boundary.append((i + (2*N)*m, disp))
    
    return boundary

def merge_boudary_conditions(bc1, bc2):
    bc = []
    for i in bc1:
        bc.append(i)
    for i in bc2:
        if i not in bc:
            bc.append(i)
    return bc

def edgeForces(F, force, dir, l, N, M=0, element_type='tri', mesh_type='coarse'):
    """
    Apply forces along the edges of the mesh.
    
    Parameters:
        F (np.ndarray): Global force vector
        force (tuple): Force vector (fx, fy)
        dir (str): Direction ('top', 'bottom', 'left', 'right')
        l (float): Length of the edge
        N (int): Number of elements in x direction
        M (int): Number of elements in y direction
        element_type (str): Element type ('tri' or 'quad')
        mesh_type (str): Mesh type ('coarse' or 'fine')
    
    Returns:
        np.ndarray: Updated force vector
    """
    if M == 0:
        M = N
    
    # Calculate number of nodes per side
    if mesh_type == 'coarse':
        n = N + 1  # nodes in x direction
        m = M + 1  # nodes in y direction
    else:  # fine mesh
        n = 2*N + 1
        m = 2*M + 1
    
    # Get number of nodes on the edge
    if mesh_type == 'coarse':
        nodes_on_edge_x = n
        nodes_on_edge_y = m
    else:
        nodes_on_edge_x = 2*N + 1
        nodes_on_edge_y = 2*M + 1
    
    # Calculate force per node
    if dir in ['top', 'bottom']:
        force_per_node = [f * l / (nodes_on_edge_x - 1) for f in force]
    else:  # left or right
        force_per_node = [f * l / (nodes_on_edge_y - 1) for f in force]
    
    # Apply forces
    if dir == 'top':
        for i in range(nodes_on_edge_x):
            node_index = i*m + (m-1)
            F[2*node_index] += force_per_node[0]
            F[2*node_index + 1] += force_per_node[1]
            
    elif dir == 'bottom':
        for i in range(nodes_on_edge_x):
            node_index = i*m
            F[2*node_index] += force_per_node[0]
            F[2*node_index + 1] += force_per_node[1]
            
    elif dir == 'left':
        for i in range(nodes_on_edge_y):
            node_index = i
            F[2*node_index] += force_per_node[0]
            F[2*node_index + 1] += force_per_node[1]
            
    elif dir == 'right':
        for i in range(nodes_on_edge_y):
            if mesh_type == 'coarse':
                node_index = i + (n-1)*m
            else:
                node_index = i + (2*N)*m
            F[2*node_index] += force_per_node[0]
            F[2*node_index + 1] += force_per_node[1]
    
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

def element_mass_matrix(nodes, elements, element_number, rho, t, element_type='tri', mesh_type='coarse', integration='gauss'):
    """
    Compute element mass matrix using either Gaussian or symbolic integration.
    
    Parameters:
        nodes, elements, element_number: Mesh data
        rho: Material density
        t: Thickness
        element_type: 'tri' or 'quad'
        mesh_type: 'coarse' or 'fine'
        integration: 'gauss' or 'symbolic'
    """
    # For linear triangles, use direct computation
    if element_type == 'tri' and mesh_type == 'coarse':
        return localMassMtrix(nodes, elements, element_number, rho, t)
    
    element_nodes = elements[element_number]
    n_nodes = len(element_nodes)
    Me = np.zeros((2*n_nodes, 2*n_nodes))
    
    if integration == 'gauss':
        # Use Gaussian integration
        gauss_points, weights = get_gauss_points(element_type)
        N, xi, eta = shape_functions(element_type, mesh_type)
        
        for point, weight in zip(gauss_points, weights):
            # Evaluate shape functions at Gauss point
            N_eval = [float(n.subs({'xi': point[0], 'eta': point[1]})) for n in N]
            
            # Calculate Jacobian at this point
            _, J, detJ = compute_B_matrix_at_point(nodes, elements, element_number, 
                                                 point[0], point[1], 
                                                 element_type, mesh_type)
            
            # Create N matrix for mass calculation
            N_matrix = np.zeros((2, 2*n_nodes))
            for i in range(n_nodes):
                N_matrix[0, 2*i] = N_eval[i]
                N_matrix[1, 2*i+1] = N_eval[i]
            
            # Add contribution to element mass matrix
            Me += weight * rho * t * N_matrix.T @ N_matrix * abs(detJ)
    
    elif integration == 'symbolic':
        # Use symbolic integration
        xi, eta = sp.symbols('xi eta')
        N, _, _ = shape_functions(element_type, mesh_type)
        
        # Calculate Jacobian symbolically
        dN_dxi, dN_deta = get_shape_derivatives(element_type, mesh_type)
        node_coords = np.array([nodes[i] for i in element_nodes])
        J = sp.zeros(2, 2)
        for i in range(n_nodes):
            J[0,0] += dN_dxi[i] * node_coords[i,0]
            J[0,1] += dN_dxi[i] * node_coords[i,1]
            J[1,0] += dN_deta[i] * node_coords[i,0]
            J[1,1] += dN_deta[i] * node_coords[i,1]
        
        detJ = sp.simplify(J.det())
        
        # Create symbolic N matrix
        N_matrix = sp.zeros(2, 2*n_nodes)
        for i in range(n_nodes):
            N_matrix[0, 2*i] = N[i]
            N_matrix[1, 2*i+1] = N[i]
        
        # Compute integrand symbolically
        integrand = N_matrix.T * N_matrix * sp.Abs(detJ) * rho * t
        
        # Perform symbolic integration
        if element_type == 'tri':
            # Integrate over unit triangle
            for i in range(2*n_nodes):
                for j in range(2*n_nodes):
                    integral = sp.integrate(
                        sp.integrate(integrand[i,j], (eta, 0, 1-xi)),
                        (xi, 0, 1)
                    )
                    Me[i,j] = float(sp.simplify(integral))
        else:
            # Integrate over unit square
            for i in range(2*n_nodes):
                for j in range(2*n_nodes):
                    integral = sp.integrate(
                        sp.integrate(integrand[i,j], (eta, -1, 1)),
                        (xi, -1, 1)
                    )
                    Me[i,j] = float(sp.simplify(integral))
    
    else:
        raise ValueError("Integration method must be either 'gauss' or 'symbolic'")
    
    return Me

def global_mass_matrix(nodes, elements, rho, t, element_type='tri', mesh_type='coarse', integration='gauss'):
    """
    Assemble the global mass matrix for the entire mesh.
    """
    n_nodes = len(nodes)
    M_global = np.zeros((2*n_nodes, 2*n_nodes))
    
    for element_number in elements:
        Me = element_mass_matrix(nodes, elements, element_number, rho, t, 
                               element_type, mesh_type, integration)
        element_nodes = elements[element_number]
        
        # Assembly
        for i in range(len(element_nodes)):
            for j in range(len(element_nodes)):
                M_global[2*element_nodes[i], 2*element_nodes[j]] += Me[2*i, 2*j]
                M_global[2*element_nodes[i]+1, 2*element_nodes[j]] += Me[2*i+1, 2*j]
                M_global[2*element_nodes[i], 2*element_nodes[j]+1] += Me[2*i, 2*j+1]
                M_global[2*element_nodes[i]+1, 2*element_nodes[j]+1] += Me[2*i+1, 2*j+1]
    
    return M_global

def normalize_matrix(matrix):
    max_value = np.max(np.abs(matrix))
    if max_value == 0:
        return matrix,max_value
    return matrix / max_value,max_value