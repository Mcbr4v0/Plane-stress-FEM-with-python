# all of the imports

import Polygones
import matplotlib.pyplot as plt
import numpy as np
import time
'''
This script compares the performance of two methods for calculating the bandwidth of a global stiffness matrix and converting it to a banded format.
It measures the time taken for each method and plots the results.
'''
# Define the material properties and the number of elements per side
E = 200e9
nu = 0.3
N = 10
t = 0.1
x = []
time_convert_to_banded = []
time_convert_to_banded_optimized = []
time_calculate_bandwidth = []
time_calculate_bandwidth_optimized = []

for N in range(1, 21):
    nodes, elements = Polygones.mesh(10, 10, N)  # create the mesh
    boundary_conditions = Polygones.boundary('left', [0, 0], N)  # create the boundary conditions, the left side is fixed

    F = np.zeros(2 * len(nodes), dtype=float)
    F = Polygones.edgeForces(F, [1e10, 0], 'right', 10, N)  # apply the force on the right side 1e10 in the x direction
    K = Polygones.global_stiffness_matrix(nodes, elements, E, nu, t)  # compute the global stiffness matrix
    K, F,constrainDofs = Polygones.apply_boundary_conditions(K, F, boundary_conditions)  # apply the boundary conditions

    # Measure time for calculate_bandwidth
    start_time = time.time()
    lower_bandwidth, upper_bandwidth = Polygones.calculate_bandwidth(K)
    end_time = time.time()
    time_calculate_bandwidth.append(end_time - start_time)

    # Measure time for calculate_bandwidth_optimized
    start_time = time.time()
    lower_bandwidth_opt, upper_bandwidth_opt = Polygones.calculate_bandwidth_optimized(K)
    end_time = time.time()
    time_calculate_bandwidth_optimized.append(end_time - start_time)

    # Measure time for convert_to_banded
    start_time = time.time()
    K_banded = Polygones.convert_to_banded(K, lower_bandwidth, upper_bandwidth)
    end_time = time.time()
    time_convert_to_banded.append(end_time - start_time)

    # Measure time for convert_to_banded_optimized
    start_time = time.time()
    K_banded_opt = Polygones.convert_to_banded_optimized(K, lower_bandwidth_opt, upper_bandwidth_opt)
    end_time = time.time()
    time_convert_to_banded_optimized.append(end_time - start_time)

    x.append(N)
    print(N)  # print the number of the iteration

# Create a figure and axis for bandwidth calculation time comparison plot
fig, ax = plt.subplots()
ax.plot(x, time_calculate_bandwidth, label='calculate_bandwidth')
ax.plot(x, time_calculate_bandwidth_optimized, label='calculate_bandwidth_optimized')
ax.legend()
ax.set_title('Comparison of bandwidth calculation time')
ax.set_xlabel("Nombre d'élément par coté")
ax.set_ylabel('Time (seconds)')

# Save the bandwidth calculation time comparison plot figure
fig.savefig("bandwidth_calculation_time_comparison.png")

# Show the bandwidth calculation time comparison plot
plt.show()

# Create a figure and axis for banded conversion time comparison plot
fig, ax = plt.subplots()
ax.plot(x, time_convert_to_banded, label='convert_to_banded')
ax.plot(x, time_convert_to_banded_optimized, label='convert_to_banded_optimized')
ax.legend()
ax.set_title('Comparison of banded conversion time')
ax.set_xlabel("Nombre d'élément par coté")
ax.set_ylabel('Time (seconds)')

# Save the banded conversion time comparison plot figure
fig.savefig("banded_conversion_time_comparison.png")

# Show the banded conversion time comparison plot
plt.show()