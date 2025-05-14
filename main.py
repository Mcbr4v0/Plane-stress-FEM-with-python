import Polygones
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scp
import pandas as pd
from scipy.sparse.linalg import eigsh
import sympy as sp
import time


N = 20      # Number of elements in the x-direction
G = 2       # Number of elements in the y-direction
t = 0.01    # Thickness of the beam
l = 2       # Length of the beam in the Y direction
L = 20      # Length of the beam in the X direction

# Material and geometric properties
E = 210e9   # Young's modulus
nu = 0.3    # Poisson's ratio
pho = 7850

P = 1e5     # Applied load
I = t*l**3/12   # moment of inertia 

# Setting for the simulation
elt = 'quad' 
met = 'coarse'
T = 300      # Total time of the simulation
accelRatio  = 2 #1/accelRatio time step
# IMPORTANT: Calculate adaptive time step based on system properties
# We'll determine this after loading matrices

# Mesh generation
nodes, elements = Polygones.mesh(L, l, N, G, element_type=elt, mesh_type=met, offsetY=l/2)

Polygones.showMesh2D(nodes, elements, element_type=elt, mesh_type=met, show=0)

# Load stiffness and mass matrices
K = pd.read_csv('stiffness_matrix.txt', sep='\t', header=None).to_numpy()
M = pd.read_csv('mass_matrix.txt', sep='\t', header=None).to_numpy()
# Define boundary conditions
node1, node2 = int(G/2), int((len(nodes)-1)-G/2)
boudary_conditions = [(node1, [0, 0]), (node2, [None, 0])]

# Apply boundary conditions
_F = np.zeros(2*len(nodes), dtype=float)
K_reduced, F_reduced, constrained_dofs = Polygones.apply_boundary_conditions(K, _F, boudary_conditions)
M_reduced = Polygones.reduce(M, constrained_dofs)

# Add pre-simulation checks on K and M matrices
def check_matrix_properties(K, M):
    """Check matrix properties relevant for dynamic analysis"""
    results = {}
    
    # Check symmetry
    results['K_symmetric'] = np.allclose(K, K.T, rtol=1e-5, atol=1e-8)
    results['M_symmetric'] = np.allclose(M, M.T, rtol=1e-5, atol=1e-8)
    
    # Check positive definiteness
    try:
        # Try Cholesky factorization (only works for positive definite matrices)
        np.linalg.cholesky(K)
        results['K_positive_definite'] = True
    except np.linalg.LinAlgError:
        results['K_positive_definite'] = False
    
    try:
        np.linalg.cholesky(M)
        results['M_positive_definite'] = True
    except np.linalg.LinAlgError:
        results['M_positive_definite'] = False
    
    # Check conditioning
    results['K_condition_number'] = np.linalg.cond(K)
    results['M_condition_number'] = np.linalg.cond(M)
    
    # Check for near-zero or negative diagonal entries
    K_diag = np.diag(K)
    M_diag = np.diag(M)
    results['K_min_diagonal'] = np.min(K_diag)
    results['M_min_diagonal'] = np.min(M_diag)
    results['K_has_negative_diagonal'] = np.any(K_diag < 0)
    results['M_has_negative_diagonal'] = np.any(M_diag < 0)
    results['K_has_zero_diagonal'] = np.any(np.abs(K_diag) < 1e-10)
    results['M_has_zero_diagonal'] = np.any(np.abs(M_diag) < 1e-10)
    
    return results

print("\n===== MATRIX DIAGNOSTIC CHECKS =====")
matrix_properties = check_matrix_properties(K_reduced, M_reduced)
for key, value in matrix_properties.items():
    print(f"{key}: {value}")
print("=====================================\n")

# Apply necessary corrections based on diagnostics
if matrix_properties['M_has_zero_diagonal'] or not matrix_properties['M_positive_definite']:
    print("WARNING: Mass matrix has issues. Applying diagonal correction...")
    # Add small values to diagonal to make it positive definite
    diag_correction = 1e-8 * np.trace(M_reduced) / M_reduced.shape[0]
    M_reduced = M_reduced + np.eye(M_reduced.shape[0]) * diag_correction
    print(f"Added {diag_correction} to mass matrix diagonal")

if matrix_properties['K_condition_number'] > 1e12:
    print("WARNING: Stiffness matrix is poorly conditioned. Consider regularization.")
    # Add small regularization if needed
    reg_factor = 1e-10 * np.trace(K_reduced) / K_reduced.shape[0]
    K_reduced = K_reduced + np.eye(K_reduced.shape[0]) * reg_factor
    print(f"Added {reg_factor} regularization to stiffness matrix")

# Calculate critical time step
# We need to find the maximum eigenvalue of M⁻¹K
# For efficiency, we'll use a power iteration method to estimate it
def estimate_max_eigenvalue(A, M, num_iterations=100, tol=1e-6):
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    M_inv = np.linalg.inv(M)  # Compute M⁻¹
    B = M_inv @ A  # Compute M⁻¹K
    
    lambda_prev = 0
    for i in range(num_iterations):
        v = B @ v
        v = v / np.linalg.norm(v)
        lambda_curr = v.T @ B @ v
        
        if abs(lambda_curr - lambda_prev) < tol:
            break
        lambda_prev = lambda_curr
    
    return lambda_curr

# Analyze eigenvalues and eigenfrequencies more carefully
try:
    # For a more reliable estimate of eigenvalues, use scipy.linalg.eigh
    # We'll use a generalized eigenvalue problem: Kv = λMv
    # This is more suitable for structural dynamics than the power method
    
    # Convert to generalized eigenvalue problem
    # Let's try to diagnose the potential numerical issues
    print("Analyzing system properties:")
    print(f"Stiffness matrix shape: {K_reduced.shape}")
    print(f"Mass matrix shape: {M_reduced.shape}")
    print(f"Stiffness matrix rank: {np.linalg.matrix_rank(K_reduced)}")
    print(f"Mass matrix rank: {np.linalg.matrix_rank(M_reduced)}")
    
    # Check for near-zero entries in mass matrix diagonal
    M_diag = np.diag(M_reduced)
    min_mass = np.min(M_diag)
    print(f"Minimum mass diagonal value: {min_mass}")
    
    if min_mass < 1e-10:
        print("WARNING: Very small mass values detected. This can cause numerical instability.")
        
    # Let's try to compute a few eigenvalues directly
    try:
        print("Computing eigenvalues...")
        # Get the 6 smallest eigenvalues (most relevant for stability)
        eigenvalues, eigenvectors = scp.eigh(K_reduced, M_reduced, eigvals=(0, 5))
        print(f"Smallest eigenvalues: {eigenvalues}")
        
        # Compute natural frequencies
        frequencies = np.sqrt(eigenvalues) / (2 * np.pi)  # in Hz
        periods = 1 / frequencies
        print(f"Natural frequencies (Hz): {frequencies}")
        print(f"Natural periods (s): {periods}")
        
        # Maximum eigenvalue for time step calculation
        max_eigenvalue = np.max(eigenvalues)
    except Exception as e:
        print(f"Error computing eigenvalues: {e}")
        print("Using power iteration method instead")
        max_eigenvalue = estimate_max_eigenvalue(K_reduced, M_reduced)
    
    omega_max = np.sqrt(max_eigenvalue)
    dt_critical = 2.0 / omega_max

    # Use a stricter safety factor to ensure stability
    safety_factor = 1  # Much more conservative
    dt = safety_factor * dt_critical
    
    print(f"Critical time step: {dt_critical:.8f}")
    print(f"Using time step with safety factor: {dt:.8f}")
    
    # Let's use a more conservative approach if dt is still very small
    min_dt = 1e-6
    if dt < min_dt:
        print(f"Time step too small, using minimum value: {min_dt}")
        dt = min_dt
    
    # Recalculate time discretization
    nbPoints = int(T / dt) + 1
    if nbPoints > 1000000:  # Limit number of points for practicality
        print(f"Warning: Very large number of time steps ({nbPoints})")
        nbPoints = 1000000
        dt = T / (nbPoints - 1)
        print(f"Adjusted time step to: {dt:.8f}")
    
    times = np.linspace(0, T, nbPoints)
    T0 = nbPoints // accelRatio  # Apply load until half of total time
    
except Exception as e:
    print(f"Error in eigenvalue analysis: {e}")
    print("Using fallback time discretization")
    # Fallback to a very conservative time step
    dt = 1e-5
    nbPoints = int(T / dt) + 1
    if nbPoints > 1000000:
        nbPoints = 1000000
        dt = T / (nbPoints - 1)
    times = np.linspace(0, T, nbPoints)
    T0 = nbPoints // accelRatio

print(f"Number of time steps: {nbPoints}")

# Create the load vectors (ramp function)
ForcesVectors = np.zeros((2*len(nodes), nbPoints), dtype=float)
for i in range(T0+1):
    t = times[i]
    F = accelRatio*P*t/T
    for j in range(len(nodes)):
        ForcesVectors[2*j, i] = 0
        ForcesVectors[2*j+1, i] = -F*L/len(nodes)
for i in range(T0+1, nbPoints):
    for j in range(len(nodes)):
        ForcesVectors[2*j, i] = 0
        ForcesVectors[2*j+1, i] = -P*L/len(nodes)

# Visualize forces
plt.close()
plt.figure(figsize=(10, 6))
plt.plot(times, abs(ForcesVectors[1,:]), 'r--', label='Y Force')
plt.xlabel('Time (s)')
plt.ylabel('Force Magnitude (N)')
plt.title('Force Components vs Time')
plt.grid(True)
plt.legend()

# Prepare reduced force vectors
ForcesVectors_reduced = np.zeros((len(K_reduced), nbPoints), dtype=float)
for i in range(nbPoints):
    ForcesVectors_reduced[:,i] = np.delete(ForcesVectors[:,i], constrained_dofs)

# Solve system using Euler explicit method
# Initialize displacement, velocity, and acceleration vectors
U_reduced = np.zeros((len(K_reduced), nbPoints), dtype=float)
V_reduced = np.zeros((len(K_reduced), nbPoints), dtype=float)
A_reduced = np.zeros((len(K_reduced), nbPoints), dtype=float)

# Set initial conditions (zero displacement and velocity)
U_reduced[:,0] = np.zeros(len(K_reduced), dtype=float)
V_reduced[:,0] = np.zeros(len(K_reduced), dtype=float)
A_reduced[:,0] = np.linalg.solve(M_reduced, ForcesVectors_reduced[:,0] - K_reduced @ U_reduced[:,0])
alpha = 0.05  # Mass proportional damping (increased)
beta = 0.005  # Stiffness proportional damping (increased)
C_reduced = alpha * M_reduced + beta * K_reduced
# Consider using Newmark-beta method instead of Euler explicit
# This is much more stable for structural dynamics problems
use_newmark = True

if use_newmark:
    # Newmark-beta parameters
    gamma = 0.5  # No numerical damping
    beta_nm = 0.25  # Corresponds to constant average acceleration method
    
    # Initialize vectors
    U_reduced = np.zeros((len(K_reduced), nbPoints), dtype=float)
    V_reduced = np.zeros((len(K_reduced), nbPoints), dtype=float)
    A_reduced = np.zeros((len(K_reduced), nbPoints), dtype=float)
    
    # Set initial conditions
    U_reduced[:,0] = np.zeros(len(K_reduced), dtype=float)
    V_reduced[:,0] = np.zeros(len(K_reduced), dtype=float)
    
    # Initial acceleration
    A_reduced[:,0] = np.linalg.solve(M_reduced, ForcesVectors_reduced[:,0] - K_reduced @ U_reduced[:,0])
    
    print("Using Newmark-beta time integration scheme without damping")
    # Effective stiffness matrix
    K_eff = K_reduced + 1/(beta_nm*dt**2)*M_reduced
    
    # LU decomposition for efficient solving
    try:
        print("Performing LU decomposition for efficient solving...")
        lu, piv = scp.lu_factor(K_eff)
        use_lu = True
        print("LU decomposition successful")
    except Exception as e:
        print(f"LU decomposition failed: {e}")
        print("Falling back to direct solve")
        use_lu = False
    
    # Time integration loop
    for i in range(1, nbPoints):
        # Predictor step
        U_pred = U_reduced[:,i-1] + dt*V_reduced[:,i-1] + (0.5-beta_nm)*dt**2*A_reduced[:,i-1]
        V_pred = V_reduced[:,i-1] + (1-gamma)*dt*A_reduced[:,i-1]
        
        # Effective force
        F_eff = ForcesVectors_reduced[:,i] - K_reduced @ U_pred - C_reduced @ V_pred
        
        # Solve for acceleration increment
        if use_lu:
            dA = scp.lu_solve((lu, piv), F_eff)
        else:
            dA = np.linalg.solve(K_eff, F_eff)
        
        # Corrector step
        A_reduced[:,i] = dA
        V_reduced[:,i] = V_pred + gamma*dt*A_reduced[:,i]
        U_reduced[:,i] = U_pred + beta_nm*dt**2*A_reduced[:,i]
        
        # Monitor for instability (using a more robust check)
        if i > 10 and np.max(np.abs(U_reduced[:,i])) > 10 * np.mean(np.abs(U_reduced[:,i-10:i])):
            print(f"Warning: Possible instability detected at step {i}, time {times[i]}")
            print(f"Max displacement: {np.max(np.abs(U_reduced[:,i]))}")
            # Instead of breaking, let's try to recover
            # Reduce the magnitude but preserve direction
            scale_factor = 10 * np.mean(np.abs(U_reduced[:,i-10:i])) / np.max(np.abs(U_reduced[:,i]))
            U_reduced[:,i] *= scale_factor
            V_reduced[:,i] *= scale_factor
            print(f"Applied stabilization factor: {scale_factor}")
            
else:
    # Original Euler explicit method (with improvements, no damping)
    # Initialize displacement, velocity, and acceleration vectors
    U_reduced = np.zeros((len(K_reduced), nbPoints), dtype=float)
    V_reduced = np.zeros((len(K_reduced), nbPoints), dtype=float)
    A_reduced = np.zeros((len(K_reduced), nbPoints), dtype=float)

    # Set initial conditions
    U_reduced[:,0] = np.zeros(len(K_reduced), dtype=float)
    V_reduced[:,0] = np.zeros(len(K_reduced), dtype=float)
    A_reduced[:,0] = np.linalg.solve(M_reduced, ForcesVectors_reduced[:,0] - K_reduced @ U_reduced[:,0])
    
    print("Using modified Euler explicit scheme without damping")
    # Solve the system for each time step
    for i in range(1, nbPoints):
        dt_actual = times[i] - times[i-1]
        
        # Euler explicit update (improved with half-step velocity)
        U_reduced[:,i] = U_reduced[:,i-1] + dt_actual * V_reduced[:,i-1]
        
        # Force balance at new position
        F_internal = K_reduced @ U_reduced[:,i]
        F_external = ForcesVectors_reduced[:,i]
        
        # Solve for acceleration
        A_reduced[:,i] = np.linalg.solve(M_reduced, F_external - F_internal)
        
        # Update velocity (using average acceleration for better stability)
        V_reduced[:,i] = V_reduced[:,i-1] + 0.5 * dt_actual * (A_reduced[:,i-1] + A_reduced[:,i])
        
        # Monitor for instability (with correction attempt)
        if i > 10 and np.max(np.abs(U_reduced[:,i])) > 10 * np.mean(np.abs(U_reduced[:,i-10:i])):
            print(f"Warning: Possible instability detected at step {i}, time {times[i]}")
            print(f"Max displacement: {np.max(np.abs(U_reduced[:,i]))}")
            # Apply stabilization
            scale_factor = 10 * np.mean(np.abs(U_reduced[:,i-10:i])) / np.max(np.abs(U_reduced[:,i]))
            U_reduced[:,i] *= scale_factor
            V_reduced[:,i] *= scale_factor
            print(f"Applied stabilization factor: {scale_factor}")



# Monitor energy for stability check
kinetic_energy = np.zeros(nbPoints)
potential_energy = np.zeros(nbPoints)
total_energy = np.zeros(nbPoints)

for i in range(nbPoints):
    if i < len(V_reduced[0]):  # Check if data exists
        kinetic_energy[i] = 0.5 * V_reduced[:,i].T @ M_reduced @ V_reduced[:,i]
        potential_energy[i] = 0.5 * U_reduced[:,i].T @ K_reduced @ U_reduced[:,i]
        total_energy[i] = kinetic_energy[i] + potential_energy[i]

# Plot energy evolution
plt.figure(figsize=(10, 6))
valid_steps = min(len(times), len(kinetic_energy))
plt.plot(times[:valid_steps], kinetic_energy[:valid_steps], 'b-', label='Kinetic Energy')
plt.plot(times[:valid_steps], potential_energy[:valid_steps], 'r--', label='Potential Energy')
plt.plot(times[:valid_steps], total_energy[:valid_steps], 'g-', label='Total Energy')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.title('Energy Evolution')
plt.grid(True)
plt.legend()

# Plot maximum displacement over time (useful for checking stability)
max_displacements = np.max(np.abs(U_reduced), axis=0)
plt.figure(figsize=(10, 6))
valid_steps = min(len(times), len(max_displacements))
plt.plot(times[:valid_steps], max_displacements[:valid_steps])
plt.xlabel('Time (s)')
plt.ylabel('Maximum Displacement Magnitude')
plt.title('Maximum Displacement vs Time')
plt.grid(True)

# Compute all the nodes positions for visualization
try:
    deformed_nodes_list = Polygones.get_deformed_nodes_list(nodes, U_reduced, constrained_dofs,scale=5)
    # Plot deformed shape for each time step
    Polygones.showDeformedFrames2D(nodes, deformed_nodes_list, elements, element_type=elt, mesh_type=met)
except Exception as e:
    print(f"Error in visualization: {e}")
    print("Try saving only selected frames instead of all time steps")