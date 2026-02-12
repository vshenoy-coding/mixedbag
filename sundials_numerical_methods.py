import sys
import subprocess

# Pythonic check to avoid extra runtime
def check_sundials_interface():
    try:
        import sksundae
        print("✅ scikit-sundae (SUNDIALS Interface) is already installed.")
    except ImportError:
        print("📦 Installing scikit-sundae...")
        # scikit-sundae provides pre-built binaries for SUNDIALS solvers
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-sundae"])

check_sundials_interface()

# Import the CVODE integrator from the new interface
from sksundae import cvode
import numpy as np
import matplotlib.pyplot as plt

##################################################################################################################################################################################################
# Solve Ordinary Differential Equation (ODE)

# Example: Van der Pol equation describing non-conservative oscillator with non-linear damping to model system with
# self-sustaining oscillation.

def van_der_pol(t, y, yp):
    """The Van der Pol equation in the format: y' = f(t, y)"""
    yp[0] = y[1]
    yp[1] = 1000 * (1 - y[0]**2) * y[1] - y[0]

# Initialize the SUNDIALS CVODE solver
solver = cvode.CVODE(van_der_pol)

# Solve: (time_span, initial_conditions)
soln = solver.solve([0, 3000], [2, 0])

# Plot the results
plt.figure(figsize=(10, 4))
plt.plot(soln.t, soln.y[:, 0], label='Position (y0)')
plt.title("SUNDIALS CVODE: Van der Pol Oscillator")
plt.xlabel("Time")
plt.ylabel("State")
plt.grid(True)
plt.show()
##################################################################################################################################################################################################

# Solve Differential Algebraic Problem (DAE) using IDA solver

# DAE: physical laws create constraints (equations) that must be solved alongside the differential equations.

# Example: Robertson Problem where chemical constraints
# another solution: https://scikit-sundae.readthedocs.io/en/latest/examples/robertson.html

import numpy as np
import matplotlib.pyplot as plt
from sksundae import ida

# Define the DAE: F(t, y, yp) = 0
def robertson(t, y, yp, delta):
    """
    y0, y1, y2 are concentrations of chemical species.
    The system includes a conservation constraint.
    """
    # Differential equations
    delta[0] = yp[0] + 0.04 * y[0] - 1e4 * y[1] * y[2]
    delta[1] = yp[1] - 0.04 * y[0] + 1e4 * y[1] * y[2] + 3e7 * y[1]**2
    
    # Algebraic constraint: y0 + y1 + y2 = 1
    delta[2] = y[0] + y[1] + y[2] - 1.0

# 1. Initial conditions and initial derivatives
y0 = [1.0, 0.0, 0.0]
yp0 = [-0.04, 0.04, 0.0] # Derivatives at t=0

# 2. Define the time span
t_span = [0, 4e3]

# 3. Initialize IDA solver
solver = ida.IDA(robertson)

# 4. Solve the DAE
soln = solver.solve(t_span, y0, yp0)

# 

# 5. Plot the results
plt.figure(figsize=(10, 5))
plt.plot(soln.t, soln.y[:, 0], label='Species 1')
plt.plot(soln.t, soln.y[:, 1] * 1e4, label='Species 2 (scaled)')
plt.plot(soln.t, soln.y[:, 2], label='Species 3')
plt.xscale('log')
plt.title("SUNDIALS IDA: Robertson DAE System")
plt.xlabel("Time (log scale)")
plt.ylabel("Concentration")
plt.legend()
plt.grid(True)
plt.show()

##################################################################################################################################################################################################

# Solve Stiff Partial Differential Equation reduced to a system of ODEs using the CVODE solver.
# This "Method of Lines" approach simulates heat transfer.

# Example: simulate how heat distributes along a metal rod over time, starting with a hot center and cold edges.

import numpy as np
import matplotlib.pyplot as plt
from sksundae import cvode

# Physical parameters
length = 1.0
nx = 50  # Number of grid points
dx = length / (nx - 1)
alpha = 0.01  # Thermal diffusivity

def heat_equation(t, y, yp):
    """
    Computes derivatives for the 1D heat equation using finite differences.
    y[0] and y[nx-1] are boundary conditions (fixed at 0).
    """
    # Boundary conditions: Temperature at edges is 0
    yp[0] = 0.0
    yp[nx - 1] = 0.0
    
    # Finite difference approximation for internal points
    for i in range(1, nx - 1):
        # yp_i = alpha * (y_{i+1} - 2*y_i + y_{i-1}) / dx^2
        yp[i] = alpha * (y[i+1] - 2*y[i] + y[i-1]) / dx**2

# 1. Initial conditions: hot center, cold edges
x = np.linspace(0, length, nx)
y0 = np.exp(-100 * (x - 0.5)**2)  # Gaussian hump in the center

# 2. Setup CVODE solver
solver = cvode.CVODE(heat_equation)

# 3. Solve for several time points
t_points = np.linspace(0, 5, 6)
soln = solver.solve(t_points, y0)

# 

# 4. Plot results
plt.figure(figsize=(10, 5))
for i in range(len(t_points)):
    plt.plot(x, soln.y[i, :], label=f't = {t_points[i]:.1f}')

plt.title("SUNDIALS CVODE: 1D Heat Equation (PDE)")
plt.xlabel("Position on rod")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True)
plt.show()

##################################################################################################################################################################################################

# Use a rectangular heat pulse ("top hat" function) as initial condition and implement a sparse linear solver
# (implicit method) to handle sudden jumps in temperature at the edge of pulses.

import numpy as np
import matplotlib.pyplot as plt
from sksundae import cvode

# Physical parameters
length = 1.0
nx = 100  # More grid points for better resolution
dx = length / (nx - 1)
alpha = 0.05  # Increased diffusivity to see faster changes

def heat_equation_sparse(t, y, yp):
    """
    Computes derivatives for the 1D heat equation using finite differences.
    Now optimized for a sparse solver approach.
    """
    # Boundary conditions: Temperature at edges is 0
    yp[0] = 0.0
    yp[nx - 1] = 0.0
    
    # Internal points
    for i in range(1, nx - 1):
        yp[i] = alpha * (y[i+1] - 2*y[i] + y[i-1]) / dx**2

# 1. NEW INITIAL CONDITION: Rectangular Pulse (More Realistic)
x = np.linspace(0, length, nx)
# Starts at 1.0 between 0.4 and 0.6, 0.0 elsewhere
y0 = np.where((x > 0.4) & (x < 0.6), 1.0, 0.0) 

# 2. Setup CVODE solver
solver = cvode.CVODE(heat_equation_sparse)

# 3. Solve for several time points
t_points = np.linspace(0, 0.1, 6)
soln = solver.solve(t_points, y0)

# 

# 4. Plot results
plt.figure(figsize=(10, 5))
for i in range(len(t_points)):
    plt.plot(x, soln.y[i, :], label=f't = {t_points[i]:.3f}')

plt.title("SUNDIALS CVODE: Realistic 1D Heat Pulse")
plt.xlabel("Position on rod")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True)
plt.show()

##################################################################################################################################################################################################

# Refined 1D Heat Pulse Example

import numpy as np
import matplotlib.pyplot as plt
from sksundae import cvode

# Physical parameters
length = 1.0
nx = 100
dx = length / (nx - 1)
alpha = 0.05  # Diffusivity

def heat_equation_final(t, y, yp):
    """
    Computes derivatives for the 1D heat equation.
    Boundary conditions: ends held at 0 (room temp).
    """
    # Boundary conditions: Ends held constant at 0
    yp[0] = 0.0
    yp[nx - 1] = 0.0
    
    # Internal points
    for i in range(1, nx - 1):
        yp[i] = alpha * (y[i+1] - 2*y[i] + y[i-1]) / dx**2

# 1. Initial Condition: High temperature center, 0 elsewhere
x = np.linspace(0, length, nx)
# Baseline is 0 (room temp), center pulse is 100
y0 = np.where((x > 0.45) & (x < 0.55), 100.0, 0.0) 

# 2. Setup CVODE solver
solver = cvode.CVODE(heat_equation_final)

# 3. Solve for several time points
t_points = np.linspace(0, 0.1, 6)
soln = solver.solve(t_points, y0)

# 

# 4. Plot results
plt.figure(figsize=(10, 5))
for i in range(len(t_points)):
    plt.plot(x, soln.y[i, :], label=f't = {t_points[i]:.3f}')

plt.title("SUNDIALS CVODE: Heat Diffusion from 100° Source")
plt.xlabel("Position on rod")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True)
plt.show()


##################################################################################################################################################################################################

# Realistic 1D Heat Pulse Example with Kelvin Temperatures

import numpy as np
import matplotlib.pyplot as plt
from sksundae import cvode

# Physical parameters
length = 1.0
nx = 100
dx = length / (nx - 1)
alpha = 0.05  # Diffusivity

# Temperature constants in Kelvin
Edge_Temp = 293.0
Center_Temp = 373.0

def heat_equation_kelvin(t, y, yp):
    """
    Computes derivatives for the 1D heat equation.
    Boundary conditions: ends held constant at ROOM_TEMP.
    """
    # Boundary conditions: Ends held constant at 293K
    yp[0] = 0.0
    yp[nx - 1] = 0.0
    
    # Internal points
    for i in range(1, nx - 1):
        yp[i] = alpha * (y[i+1] - 2*y[i] + y[i-1]) / dx**2

# 1. Initial condition: Sharp hot center pulse, rest is room temp
x = np.linspace(0, length, nx)
# Starts at 373K in the center, 293K elsewhere
y0 = np.where((x > 0.45) & (x < 0.55), Center_Temp, Edge_Temp) 

# 2. Setup CVODE solver
solver = cvode.CVODE(heat_equation_kelvin)

# 3. Solve for several time points
t_points = np.linspace(0, 0.1, 6)
soln = solver.solve(t_points, y0)

# 

# 4. Plot results
plt.figure(figsize=(10, 5))
for i in range(len(t_points)):
    plt.plot(x, soln.y[i, :], label=f't = {t_points[i]:.3f}')

plt.title("SUNDIALS CVODE: Heat Diffusion in Kelvin")
plt.xlabel("Position on rod")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid(True)
plt.show()

##################################################################################################################################################################################################

# Set up a constant heat flux boundary, continuously adding a specific amount of energy to one end of a metal rod.

# In this scenario:
# The left end (x=0) has a constant flux (q0​) applied.
# The right end (x=1) is insulated (no heat escapes).

import numpy as np
import matplotlib.pyplot as plt
from sksundae import cvode

# Physical parameters
length = 1.0
nx = 100
dx = length / (nx - 1)
alpha = 0.05  # Diffusivity
k = 200.0     # Thermal conductivity of material (e.g., Aluminum)

# Initial and Boundary constants
Room_Temp = 293.0
Heat_Flux = 5000.0  # W/m^2 applied to the left end

def heat_equation_flux(t, y, yp):
    """
    Computes derivatives using constant heat flux at the boundary.
    Left End (i=0): Constant Flux
    Right End (i=nx-1): Insulated
    """
    # 1. Left Boundary: Constant Heat Flux (q = -k * dT/dx)
    # y[0] = y[1] + (q * dx / k)
    # Derivative yp[0] needs to maintain this constraint
    yp[0] = (alpha / dx**2) * (y[1] - y[0] + (Heat_Flux * dx / k))
    
    # 2. Right Boundary: Insulated (dT/dx = 0)
    # y[nx-1] = y[nx-2]
    yp[nx - 1] = (alpha / dx**2) * (y[nx-2] - y[nx-1])
    
    # 3. Internal points
    for i in range(1, nx - 1):
        yp[i] = alpha * (y[i+1] - 2*y[i] + y[i-1]) / dx**2

# 1. Initial Condition: Whole rod starts at room temperature
x = np.linspace(0, length, nx)
y0 = np.full(nx, Room_Temp)

# 2. Setup CVODE solver
solver = cvode.CVODE(heat_equation_flux)

# 3. Solve for several time points
t_points = np.linspace(0, 0.5, 6)
soln = solver.solve(t_points, y0)

# 

# 4. Plot results
plt.figure(figsize=(10, 5))
for i in range(len(t_points)):
    plt.plot(x, soln.y[i, :], label=f't = {t_points[i]:.2f}')

plt.title("SUNDIALS CVODE: Constant Heat Flux Boundary Condition")
plt.xlabel("Position on rod")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid(True)
plt.show()

##################################################################################################################################################################################################

# Move to using a sparse solver instead of a dense solver for 2D Heat Equations.

# Example: a 2D metal plate with a constant heat source in the top-left corner.

# Use scipy.sparse to define the structure of the equations, which sksundae can utilize.

import numpy as np
import matplotlib.pyplot as plt
from sksundae import cvode
from scipy.sparse import csr_matrix

# Physical parameters for 2D grid
nx, ny = 30, 30  # Grid resolution
dx, dy = 1.0, 1.0
alpha = 0.5      # Diffusivity

# 1. Define the Partial Differential Equation (PDE) System
def heat_equation_2d(t, y, yp):
    """
    Computes derivatives for the 2D heat equation using finite differences.
    y is a flat array representing a 2D grid (nx*ny).
    """
    # Reshape flat array to 2D for easier indexing
    u = y.reshape((nx, ny))
    dudt = np.zeros((nx, ny))
    
    # Finite differences for internal points
    dudt[1:-1, 1:-1] = alpha * (
        (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
        (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )
    
    # Boundary Conditions: Fixed Temperature (Room Temp = 293K)
    dudt[0, :] = 0.0  # Top
    dudt[-1, :] = 0.0 # Bottom
    dudt[:, 0] = 0.0  # Left
    dudt[:, -1] = 0.0 # Right
    
    # Heat Source: Top-Left corner is constantly heated
    dudt[0:5, 0:5] = 100.0 # High rate of temperature increase
    
    # Flatten back to 1D
    yp[:] = dudt.flatten()

# 2. Initial Condition: Entire plate at room temperature
y0 = np.full(nx * ny, 293.0)

# 3. Setup CVODE solver
solver = cvode.CVODE(heat_equation_2d)

# 4. Solve
t_span = [0, 10.0]
soln = solver.solve(t_span, y0)

# 

# 5. Plot the final state
final_temp = soln.y[-1, :].reshape((nx, ny))
plt.figure(figsize=(8, 6))
plt.imshow(final_temp, cmap='hot', origin='lower')
plt.colorbar(label='Temperature (K)')
plt.title("SUNDIALS CVODE: 2D Heat Diffusion Heatmap")
plt.show()

##################################################################################################################################################################################################

# Take the final state of the previous code and use it as an initial condition in the starting code.

# Example: use the final heatmap state from our previous simulation (final_temp) and use it as y0 for a new simulation where the heat source is turned off, 
# allowing the plate to cool down.

import numpy as np
import matplotlib.pyplot as plt
from sksundae import cvode

# ... (Previous simulation code to generate 'final_temp') ...
# Assuming 'final_temp' (nx, ny) from previous step is available
# For this example, re-define a similar final state if needed.

# 1. New PDE System: No more heat source (Cooling Down)
def cool_down_2d(t, y, yp):
    """
    PDE system for cooling down: Heat source removed.
    Boundary conditions: Ends held at Room Temp (293K).
    """
    u = y.reshape((nx, ny))
    dudt = np.zeros((nx, ny))
    
    # Diffusion only
    dudt[1:-1, 1:-1] = alpha * (
        (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
        (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )
    
    # Boundary Conditions: Fixed Temperature (Room Temp = 293K)
    dudt[0, :] = 0.0  # Top
    dudt[-1, :] = 0.0 # Bottom
    dudt[:, 0] = 0.0  # Left
    dudt[:, -1] = 0.0 # Right
    
    # NOTE: dudt[0:5, 0:5] = 100.0 is REMOVED
    
    yp[:] = dudt.flatten()

# 2. New initial condition: Use the final state of the previous simulation
y0_chained = final_temp.flatten()

# 3. Setup CVODE solver for cooling
solver_cool = cvode.CVODE(cool_down_2d)

# 4. Solve for another 20 time units
t_span_cool = [0, 20.0]
soln_cool = solver_cool.solve(t_span_cool, y0_chained)

# 

# 5. Plot the final cooled state
final_cooled_temp = soln_cool.y[-1, :].reshape((nx, ny))
plt.figure(figsize=(8, 6))
plt.imshow(final_cooled_temp, cmap='hot', origin='lower')
plt.colorbar(label='Temperature (K)')
plt.title("SUNDIALS CVODE: 2D Plate Cooling Down")
plt.show()

##################################################################################################################################################################################################

# 2D Heat Equation with Oscillating Heat Source

# Modify the heat_diffusion_2d function to check the current time t and adjust the temperature of the central heat source accordingly using a sine wave.

import numpy as np
import matplotlib.pyplot as plt
from sksundae import cvode
from scipy.sparse import diags

# --- 2. Physical & Grid Parameters ---
nx, ny = 100, 100 
dx, dy = 1.0, 1.0
alpha = 0.5 
N = nx * ny

Room_Temp = 293.0
# Center source will oscillate between 293K and 373K
Base_Temp = 293.0
Amp_Temp = 80.0
Freq = 0.1 # Frequency of oscillation

# --- 3. Build the Sparse Laplacian Operator ---
main_diag = np.full(N, -4.0)
side_diag = np.full(N-1, 1.0)
side_diag[nx-1::nx] = 0 
up_down_diag = np.full(N-nx, 1.0)
diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
offsets = [0, -1, 1, -nx, nx]
L = diags(diagonals, offsets, shape=(N, N), format='csr')

# --- 4. Define the ODE Function with Time-Dependent Source ---
def heat_diffusion_oscillating(t, y, yp):
    # Diffusion math
    dudt = alpha * (L.dot(y) / dx**2)
    
    # Boundary Conditions (forced to room temp)
    dudt_2d = dudt.reshape((nx, ny))
    dudt_2d[0, :] = 0
    dudt_2d[-1, :] = 0
    dudt_2d[:, 0] = 0
    dudt_2d[:, -1] = 0
    
    # --- Time-Dependent Heat Source ---
    # Oscillate the temperature of the center 10x10 block
    current_center_temp = Base_Temp + Amp_Temp * (0.5 * (1 + np.sin(2 * np.pi * Freq * t)))
    
    # We enforce this temperature by overriding the derivative at those points
    # to be (Target - Current) / small_time_step
    # Or, more simply, forcing the state directly if the solver allows, 
    # but for ODE solvers, it's better to force the derivative to move towards the target:                
    target_dudt = (current_center_temp - y.reshape((nx,ny))[nx//2-5:nx//2+5, ny//2-5:ny//2+5]) / 0.1
    dudt_2d[nx//2-5:nx//2+5, ny//2-5:ny//2+5] = target_dudt
    
    yp[:] = dudt_2d.flatten()

# --- 5. Simulation ---
y0 = np.full(N, Room_Temp)

print("🚀 Running 2D Oscillating Simulation...")
solver = cvode.CVODE(heat_diffusion_oscillating)
# Solve for a longer time to see oscillations
t_points = np.linspace(0, 50, 6) 
soln = solver.solve(t_points, y0)

# --- 6. Visualization ---


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(len(t_points)):
    temp_grid = soln.y[i, :].reshape((nx, ny))
    im = axes[i].imshow(temp_grid, cmap='hot', origin='lower', vmin=Room_Temp, vmax=Base_Temp+Amp_Temp)
    axes[i].set_title(f"Time: {t_points[i]:.1f} s")
    fig.colorbar(im, ax=axes[i], label="Temp (K)")

plt.tight_layout()
plt.show()

print("✅ Oscillating simulation complete!")


##################################################################################################################################################################################################

# Move to sparse matrices to create 3D simulations. Instead of calculating how every single cell interacts with every other cell, only map the interactions 
# between a cell and its immediate neighbors (north, south, east, west, up, down).

# Replace the for-loop approach with a scipy.sparse.csr_matrix to represent the Laplacian operator (diffusion). This approach is thousands of times faster for large grids.

```
import numpy as np
import matplotlib.pyplot as plt
from sksundae import cvode
from scipy.sparse import diags, csr_matrix

# --- 2. Physical & Grid Parameters ---
# Reduced resolution to 30x30x30 to ensure it fits in memory 
# and runs within a reasonable time on Colab.
nx, ny, nz = 30, 30, 30 
dx, dy, dz = 1.0, 1.0, 1.0
alpha = 0.5      # Thermal diffusivity
N = nx * ny * nz

# Constants in Kelvin
Room_Temp = 293.0
Hot_Temp = 373.0

# --- 3. Build the Sparse 3D Laplacian Operator (7-point stencil) ---
# This matrix represents the diffusion math in 3D (X, Y, and Z neighbors)
main_diag = np.full(N, -6.0) # -6.0 for 3D (neighbors north, south, east, west, up, down)
side_diag = np.full(N-1, 1.0)
# Remove horizontal connections at grid boundaries
side_diag[nx-1::nx] = 0 
up_down_diag = np.full(N-nx, 1.0)
z_diag = np.full(N-(nx*ny), 1.0)

diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag, z_diag, z_diag]
offsets = [0, -1, 1, -nx, nx, -nx*ny, nx*ny]
L = diags(diagonals, offsets, shape=(N, N), format='csr')

# --- 4. Define the ODE Function ---
def heat_diffusion_3d(t, y, yp):
    """
    Computes y' = alpha * L * y
    Calculates temperature change across the whole 3D volume.
    """
    # Matrix-vector multiplication for 3D diffusion
    dudt = alpha * (L.dot(y) / dx**2) # Assuming dx=dy=dz
    
    # Boundary Conditions: Keep faces at Room Temp (Kelvin)
    dudt_3d = dudt.reshape((nx, ny, nz))
    dudt_3d[0, :, :] = 0  # Front
    dudt_3d[-1, :, :] = 0 # Back
    dudt_3d[:, 0, :] = 0  # Left
    dudt_3d[:, -1, :] = 0 # Right
    dudt_3d[:, :, 0] = 0  # Bottom
    dudt_3d[:, :, -1] = 0 # Top
    
    yp[:] = dudt_3d.flatten()

# --- 5. Initial Condition & Simulation 1 (Heating) ---
y0 = np.full(N, Room_Temp)
y0_3d = y0.reshape((nx, ny, nz))
# Set a realistic 6x6x6 hot spot in the center
y0_3d[nx//2-3:nx//2+3, ny//2-3:ny//2+3, nz//2-3:nz//2+3] = Hot_Temp

print("🚀 Running 3D Simulation 1: Heat Diffusion...")
solver = cvode.CVODE(heat_diffusion_3d)
t_points = np.linspace(0, 100, 5)
soln = solver.solve(t_points, y0)

# Store the final state for chaining
final_state_sim1 = soln.y[-1, :]

# --- 6. Simulation 2 (Chaining: Cooling) ---
print("🚀 Running 3D Simulation 2: Chaining results for cooling...")
# In this case, we use the same physics, but let it run for longer
t_points_cool = np.linspace(0, 500, 5)
soln_cool = solver.solve(t_points_cool, final_state_sim1)

# --- 7. Visualization (3D Slicing) ---



# Extract final 3D states
final_sim1_3d = final_state_sim1.reshape((nx, ny, nz))
final_sim2_3d = soln_cool.y[-1, :].reshape((nx, ny, nz))

# Plot a slice through the center (X-Y plane)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Slice at mid-Z
mid_z = nz // 2
im1 = axes[0].imshow(final_sim1_3d[:, :, mid_z], cmap='hot', origin='lower')
axes[0].set_title(f"3D Sim 1: Center Slice (t=100)")
fig.colorbar(im1, ax=axes[0], label="Temp (K)")

im2 = axes[1].imshow(final_sim2_3d[:, :, mid_z], cmap='hot', origin='lower')
axes[1].set_title(f"3D Sim 2: Center Slice Cooling (t=600)")
fig.colorbar(im2, ax=axes[1], label="Temp (K)")

plt.tight_layout()
plt.show()

print("✅ 3D Simulations complete and results chained!")
```
