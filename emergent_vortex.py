import sys
import subprocess

# Pythonic code to check if packages are installed ---
def ensure_dependencies():
    for pkg in ["numpy", "matplotlib"]:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure_dependencies()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. The Non-Hydrostatic Manifold ---
num_particles = 450
dt = 0.04
brane_tension = 6.0      # (Sigma) Metric stiffness; resists non-hydrostatic acceleration
pressure_perturbation = 0.9 # Localized dip in the 4D manifold (Non-hydrostatic PGF)
vorticity_coeff = 1.4    # Rotational intensity of the emergent flow

# Initialize "Seeds" (Stochastic fluctuations in the Bulk)
pos = np.random.uniform(-5, 5, (num_particles, 2))
vel = np.random.normal(0, 0.1, (num_particles, 2))

# --- 2. The Dynamics: Accelerations over Balance ---
def update(frame):
    global pos, vel

    # Metric distance from the pressure perturbation center
    dist_sq = np.sum(pos**2, axis=1, keepdims=True)
    dist = np.sqrt(dist_sq) + 0.1 

    # Force A: Non-Hydrostatic Pressure Gradient Force (PGF)
    # This isn't balanced by gravity; it's a direct metric acceleration.
    inward_dir = -pos / dist
    accel_pgf = inward_dir * (pressure_perturbation / (dist + 0.5))

    # Force B: Local Vorticity
    # The tangential component of the non-linear manifold flow
    accel_rot = np.column_stack([-pos[:, 1], pos[:, 0]]) / dist
    accel_rot *= vorticity_coeff

    # Velocity Update: Summing non-hydrostatic accelerations
    vel += (accel_pgf + accel_rot) * dt

    # --- Non-Linear Brane Damping ---
    # As the system departs from equilibrium, "Metric Tension" provides resistance.
    # High-velocity localized flows encounter topological damping.
    speed_sq = np.sum(vel**2, axis=1, keepdims=True)
    dynamic_resistance = 1.0 / (1.0 + (speed_sq / (2 * brane_tension)))
    vel *= dynamic_resistance 

    pos += vel * dt

    # The "Bulk" Cycle: Entropy dissipation and re-seeding
    mask = (dist.flatten() > 6) | (dist.flatten() < 0.05)
    pos[mask] = np.random.uniform(-5, 5, (np.sum(mask), 2))
    vel[mask] = np.random.normal(0, 0.1, (np.sum(mask), 2))

    # Update Visualization
    scat.set_offsets(pos)
    # Color particles by speed to visualize the "Criticality" of the core
    scat.set_array(np.sqrt(speed_sq.flatten()))
    return scat,

# --- 3. Rendering the Emergence and Visualizing the Attractor ---
fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.axis('off')

scat = ax.scatter(pos[:, 0], pos[:, 1], s=4, c='white', cmap='magma', alpha=0.7)

ani = FuncAnimation(fig, update, frames=200, interval=25, blit=True)
plt.show()


import sys
import subprocess

# Pythonic code to check if packages are installed ---
def ensure_dependencies():
    for pkg in ["numpy", "matplotlib"]:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure_dependencies()


import numpy as np
import matplotlib.pyplot as plt
import time

def generate_emergent_vortex():
    # 1. The Bulk Seed: Extracting unique data from the temporal Bulk
    # Each run samples a different "manifold state."
    # We use the system time to ensure the "Soil" is never the same twice.
    random_seed = int(time.time() * 1000) % 2**32
    np.random.seed(random_seed)

    grid_size = 60
    x, y = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))

    # 2. Metric Distortion: The center of the "Attractor" shifts slightly
    # Simulating a dynamic, non-stable pressure perturbation.
    # This prevents the "Perfect Center" repetition.
    center_offset_x = np.random.uniform(-0.1, 0.1)
    center_offset_y = np.random.uniform(-0.1, 0.1)

    # 3. Non-Hydrostatic Noise Floor: Stochastic interference
    # Replaces static pressure with dynamic, high-frequency fluctuations.
    # We use a fractal-like noise approach.
    noise = np.random.normal(0, 0.08, (grid_size, grid_size))
    # The pressure field dictates the "dip" in the brane
    pressure = -((x - center_offset_x)**2 + (y - center_offset_y)**2) + noise

    # 4. Emergent Flow Parameters: "Spin" and "Inflow"
    spin_strength = np.random.uniform(3.5, 6.0)
    inflow_strength = np.random.uniform(0.2, 0.6)

    # 5. Dynamics: Non-linear Flow Calculation
    # We take the gradient of the noisy pressure field to find accelerations
    v_grad, u_grad = np.gradient(pressure)
    
    # Rotational flow (tangential) + Inflow (radial)
    u = -v_grad * spin_strength - (x - center_offset_x) * inflow_strength
    v = u_grad * spin_strength - (y - center_offset_y) * inflow_strength

    magnitude = np.sqrt(u**2 + v**2)

    # 6. Rendering: Capturing the singular "Snapshot" of the manifold
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')

    # Streamplot maps the vector field across the "Manifold"
    # We vary the density and linewidth slightly to reflect the "Atmospheric State".
    ax.streamplot(x, y, u, v, color=magnitude, cmap='inferno',
                  density=np.random.uniform(1.2, 2.0),
                  linewidth=np.random.uniform(0.8, 1.6))

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('emergent_vortex.png', facecolor='black')
    plt.show()

generate_emergent_vortex()


import sys
import subprocess

# Pythonic code to check if packages are installed ---
def ensure_dependencies():
    for pkg in ["numpy", "matplotlib"]:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure_dependencies()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def generate_3d_vortex():
    # 1. Temporal Bulk Sampling: Seed the "Soil"
    random_seed = int(time.time() * 1000) % 2**32
    np.random.seed(random_seed)

    # 2. 3D Manifold Grid: Setting up the spatial domain
    grid_res = 15
    x, y, z = np.meshgrid(np.linspace(-2, 2, grid_res),
                         np.linspace(-2, 2, grid_res),
                         np.linspace(0, 5, grid_res)) # Z is height

    # 3. Dynamic Attractor: Non-Hydrostatic Center Perturbation
    # The center can tilt or drift slightly in 3D space.
    tilt_x = np.random.uniform(-0.15, 0.15)
    tilt_y = np.random.uniform(-0.15, 0.15)
    spin_power = np.random.uniform(2.0, 5.0)
    lift_power = np.random.uniform(1.5, 3.5)
    
    # Brane Tension (Sigma) - Resists high kinetic energy
    brane_tension = 4.0 

    # 4. Calculating the Manifold: Non-Linear Flow Field
    # Inflow (Radial/Tangential interaction causing convergence)
    u = -y * spin_power - (x - tilt_x*z) * 0.5
    v =  x * spin_power - (y - tilt_y*z) * 0.5
    
    # Non-Hydrostatic Updraft (Z): 
    # Convergence drives vertical acceleration, forming the funnel.
    # Updraft (Z) - Strength increases with height then tapers (the "funnel" effect).
    dist_from_center = np.sqrt((x-tilt_x*z)**2 + (y-tilt_y*z)**2)
    w = lift_power * (1.0 / (dist_from_center + 0.5))

    # Adding "Stochastic Seeds" (Quantum Noise) to trigger turbulence
    u += np.random.normal(0, 0.2, u.shape)
    v += np.random.normal(0, 0.2, v.shape)
    w += np.random.normal(0, 0.1, w.shape)
    
    # Non-Linear Brane Damping ---
    # Resistance scales with energy density, preventing unrealistically high velocities.
    speed_sq = u**2 + v**2 + w**2
    dynamic_resistance = 1.0 / (1.0 + (speed_sq / (2 * brane_tension)))
    
    u *= dynamic_resistance
    v *= dynamic_resistance
    w *= dynamic_resistance

    # 5. Rendering: Visualizing the Emergent System
    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Color mapping by velocity magnitude (Kinetic energy density)
    mag = np.sqrt(u**2 + v**2 + w**2)
    
    # 3D Quiver plot: Displays the "bones" of the emergent flow field structure 
    ax.quiver(x, y, z, u, v, w, length=0.3,
              color=plt.cm.magma(mag.flatten()/mag.max()),
              alpha=0.6, normalize=True)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig('emergent_3d_vortex.png', facecolor='black')
    plt.show()

generate_3d_vortex()
