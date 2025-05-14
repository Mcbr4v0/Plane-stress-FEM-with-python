
# Plane Stress FEM Analysis with Python

## Overview
This project implements a Finite Element Method (FEM) solver for plane stress analysis using Python. It provides tools for analyzing the mechanical behavior of 2D structures under various loading conditions.

## Author
Arthur Boudehent  
Developed during an internship at La Sapienza University, Rome

## Features
- 2D finite element analysis
- Dynamic time integration using:
  - Newmark-β method
  - Modified Euler explicit method
- Support for different element types:
  - Triangular elements (coarse/fine mesh)
  - Quadrilateral elements (coarse/fine mesh)
- Modal analysis capabilities:
  - Natural frequency calculation
  - Mode shape visualization
  - Interactive mode shape animations
- Matrix diagnostics:
  - Condition number checking
  - Positive definiteness verification
  - Symmetry validation
- Visualization tools:
  - Interactive mesh viewer
  - Dynamic deformation display
  - Energy evolution plots
  - Force visualization

## Dependencies
- NumPy: Numerical computations
- SciPy: Linear algebra and sparse matrix operations
- Matplotlib: Visualization and animations
- Pandas: Matrix I/O and data handling
- SymPy: Symbolic computations

## Project Structure
- `main.py`: Dynamic analysis implementation
- `Polygones.py`: Core FEM functions
  - Mesh generation
  - Matrix assembly
  - Shape functions
  - Visualization tools
- `TestBeam`: Static analysis examples
  - `StaticTest`: Other static analisis and comparison with analytic comparison
  -`BeamExemple.py`:
- `freeModes.py`: Modal analysis tools
- `comparison`: Comparison of numerical methods
## Usage
Basic example for running a modal analysis:
```python
import Polygones
import numpy as np

# Define material properties
E = 210e9  # Young's modulus
nu = 0.3   # Poisson's ratio
rho = 7850 # Density

# Create mesh
nodes, elements = Polygones.mesh(L=10, l=1, N=40)

# Assemble matrices
K = Polygones.global_stiffness_matrix(nodes, elements, E, nu, h)
M = Polygones.global_mass_matrix(nodes, elements, rho, h)

# Apply boundary conditions and solve