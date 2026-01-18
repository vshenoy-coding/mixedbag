# Figure 1. Thermodynamic Sounding a) and Hodographic Schematic b)

import numpy as np
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units
from metpy.plots import SkewT, Hodograph
from wk_profiles import wk_sounding, find_wk_params

# ==========================================
# 1. GENERATE THERMODYNAMICS (Panel A)
# ==========================================
heights = np.arange(0, 20001, 100)
# Base WK82 parameters from the study
params = find_wk_params(tsfc=300.0, psfc=101325.0, sbcape=2500, sblcl=1000)
prof_gen = wk_sounding(**params)
temp, dewp, pres = prof_gen(heights)

# ==========================================
# 2. GENERATE WIND PROFILES (Panel B)
# ==========================================
def create_wind_profile(z, alpha_deg, ll_mag=10.0, total_shear=22.5):
    """
    Creates u, v components based on the alpha orientation.
    0.5km is set as the origin (0,0) for visualization purposes.
    """
    u = np.zeros_like(z)
    v = np.zeros_like(z)
    alpha_rad = np.deg2rad(alpha_deg)
    
    # Calculate 0km position relative to 0.5km origin
    u_0 = -ll_mag * np.cos(alpha_rad)
    v_0 = -ll_mag * np.sin(alpha_rad)
    
    # Constant shear from 0.5 to 3km along the x-axis
    u_3 = total_shear + u_0
    
    for i, h in enumerate(z):
        if h <= 500: # 0 to 0.5km (LL Shear)
            u[i] = u_0 + (0 - u_0) * (h / 500)
            v[i] = v_0 + (0 - v_0) * (h / 500)
        elif h <= 3000: # 0.5 to 3km (Deep Shear)
            u[i] = 0 + (u_3 - 0) * ((h - 500) / 2500)
            v[i] = 0
        else: # Above 3km
            u[i] = u_3
            v[i] = 0
    return u, v

# ==========================================
# 3. VISUALIZATION
# ==========================================
fig = plt.figure(figsize=(14, 6))

# --- Panel A: Skew-T ---
skew = SkewT(fig, rotation=45, subplot=(1, 2, 1))
skew.plot(pres * units.Pa, (temp * units.K).to('degC'), 'r', linewidth=2, label='Temp')
skew.plot(pres * units.Pa, (dewp * units.K).to('degC'), 'b', linewidth=2, label='Dewpoint')
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-20, 40)
skew.ax.set_title("a) Thermodynamic Sounding (WK82)", loc='left', fontweight='bold')
skew.ax.legend()

# --- Panel B: Custom Hodograph ---
ax_hodo = fig.add_subplot(1, 2, 2)
alphas = {180: 'blue', 150: 'green', 120: 'black', 90: 'orange', 60: 'red'}

# Draw Background Rings
for r in [5, 10, 15, 20, 25]:
    circle = plt.Circle((0, 0), r, color='gray', fill=False, alpha=0.2)
    ax_hodo.add_artist(circle)
    ax_hodo.text(r, 0.5, f"{r}", color='gray', fontsize=8)

for alpha, color in alphas.items():
    u, v = create_wind_profile(heights, alpha)
    # Plot the 0-3km segments
    mask = heights <= 3000
    ax_hodo.plot(u[mask], v[mask], color=color, linewidth=2, alpha=0.8)
    
    # Mark specific heights
    ax_hodo.scatter(u[0], v[0], color=color, s=100, edgecolors='k', zorder=5) # 0km
    ax_hodo.scatter(u[heights==500], v[heights==500], color='gray', s=50, zorder=5) # 0.5km
    ax_hodo.scatter(u[heights==3000], v[heights==3000], color=color, s=100, edgecolors='k', zorder=5) # 3km
    
    # Label alpha angles
    ax_hodo.text(u[0]-2, v[0], f"α{alpha}", color=color, fontweight='bold', ha='right')

ax_hodo.set_aspect('equal')
ax_hodo.set_xlim(-25, 30)
ax_hodo.set_ylim(-15, 15)
ax_hodo.axhline(0, color='k', alpha=0.2)
ax_hodo.axvline(0, color='k', alpha=0.2)
ax_hodo.set_title("b) Hodograph Schematic ($u-v$ space)", loc='left', fontweight='bold')
ax_hodo.set_xlabel("u (m/s)")
ax_hodo.set_ylabel("v (m/s)")

plt.tight_layout()
plt.show()


