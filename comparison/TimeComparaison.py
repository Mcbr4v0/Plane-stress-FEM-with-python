# all of the imports

import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import time
'''
This script compares the performance of two methods for solving a linear system of equations:
1. Using `np.linalg.solve`
2. Using `scipy.linalg.solve_banded` with a banded matrix representation.
It measures the time taken for each method and plots the results.
It also computes the relative error between the two methods.
It works like the BandedComparaison.py script but uses the Polygones module to create the mesh and apply boundary conditions.
It is used to verify the correctness of the Polygones module and to test the performance of the two methods.
'''
#Define the material properties and the number of elements per side
E = 200e9
nu = 0.3
N = 10
t = 0.1
ec = []
x = []
time_np_solve = []
time_scipy_solve_banded = []
time_compute_K = []

def moy(U,Uprime):#function to compute the error between the two methods
    moy = 0
    for i in range(len(U)):
            moy += abs((U[i]-Uprime[i]))
    return moy/len(U)



for N in range(1, 31):
    start_time = time.time()
    nodes, elements = Polygones.mesh(10, 10, N)  # create the mesh
    boundary_conditions = Polygones.boundary('left', [0, 0],N)  # create the boundary conditions, the left side is fixed

    F = np.zeros(2 * len(nodes), dtype=float)
    F = Polygones.edgeForces(F, [1e10, 0], 'right',10,N)  # apply the force on the right side 1e10 in the x direction
    K = Polygones.global_stiffness_matrix(nodes, elements, E, nu,t)  # compute the global stiffness matrix
    K, F,constrainedDofs = Polygones.apply_boundary_conditions(K, F, boundary_conditions)  # apply the boundary conditions
    end_time = time.time()
    time_compute_K.append(end_time -start_time)  # measure the time to compute the global stiffness
    

    # Measure time for np.linalg.solve
    start_time = time.time()
    Uprime = np.linalg.solve(K, F)
    end_time = time.time()
    time_np_solve.append(end_time - start_time)

    # Measure time for scipy.linalg.solve_banded
    start_time = time.time()
    lower_bandwidth, upper_bandwidth = Polygones.calculate_bandwidth_optimized(K)  # calculate the bandwidth
    K_banded = Polygones.convert_to_banded_optimized(K, lower_bandwidth, upper_bandwidth)
    U = scipy.linalg.solve_banded((lower_bandwidth, upper_bandwidth), K_banded, F)
    end_time = time.time()
    time_scipy_solve_banded.append(end_time - start_time)

    ec.append(moy(U, Uprime))#add result to a list
    x.append(N)

    print(N) #print the number of the iteration

# Create a figure and axis for error plot
fig, ax = plt.subplots()
ax.plot(x, ec, label='Relative Error')
ax.legend()
ax.set_title('Relative Error between Absolute Method and Gauss Method')
ax.set_xlabel("Number of elements per side")
ax.set_ylabel('Error in %')

# Save the error plot figure
fig.savefig("convergence.png")

# Show the error plot
plt.show()

# Create a figure and axis for time comparison plot
fig, ax = plt.subplots()
ax.plot(x, time_np_solve, label='np.linalg.solve')
ax.plot(x, time_scipy_solve_banded, label='scipy.linalg.solve_banded')
ax.plot(x, time_compute_K, label='compute K')
ax.legend()
ax.set_title('Comparison of computation time')
ax.set_xlabel("Number of elements per side")
ax.set_ylabel('Time (seconds)')

# Save the time comparison plot figure
fig.savefig("time_comparison.png")

# Show the time comparison plot
plt.show()