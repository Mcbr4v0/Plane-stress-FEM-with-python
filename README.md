
# Plane Stress FEM Analysis with Python

## Overview
This project implements a Finite Element Method (FEM) solver for plane stress analysis using Python. It provides tools for analyzing the mechanical behavior of 2D structures under various loading conditions.

## Author
Arthur Boudehent  
Developed during an internship at La Sapienza University, Rome

## Features
- 2D mesh generation for rectangular domains
- Linear triangular elements implementation
- Plane stress analysis capabilities
- Natural frequency and mode shape analysis
- Visualization tools for:
  - Mesh generation
  - Deformed structures
  - Mode shape animations
- Boundary condition handling
- Force application along edges

## Dependencies
- NumPy: For numerical computations
- Matplotlib: For visualization and animations
- SciPy: For sparse matrix operations and linear algebra
- Pandas: For data handling, espcially for big matrices

## Structure
- `Polygones.py`: Core implementation containing FEM functions
  - Mesh generation
  - Shape functions
  - Stiffness matrix assembly
  - Mass matrix assembly
  - Boundary condition application
  - Visualization functions
- `BeamExemple.py`: Example implementation for beam analysis
  - Modal analysis
  - Natural frequency calculation
  - Mode shape visualization

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
nodes, elements = Polygones.mesh(L=10, l=1, N=40, M=8)

# Assemble matrices
K = Polygones.global_stiffness_matrix(nodes, elements, E, nu, h)
M = Polygones.global_mass_matrix(nodes, elements, rho, h)

# Apply boundary conditions and solve