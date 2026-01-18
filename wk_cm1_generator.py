# Study 
# Influence of Low-Level Shear Orientation and Magnitude on the Evolution and Rotation of Idealized Squall Lines. 
# Part I: Storm Morphology and Bulk Updraft/Mesovortex Attributes in: Monthly Weather Review Volume 152 Issue 9 (2024)
# https://journals.ametsoc.org/view/journals/mwre/152/9/MWR-D-23-0262.1.xml

# A WK82 profile is analytic (defined by equations) 
# It defines:

# A temperature profile based on a specific lapse rate.
# A moisture profile that decreases with height.
# A wind profile (usually a "half-circle" or "linear" hodograph).

# The authors modulated the standard WK82 profile by adjusting the mixing ratio.

# Credit goes to Tim Supinie (https://github.com/tsupinie/wk-profiles) for the python package wk-profiles used to modulate Weisman–Klemp thermodynamic profiles used to initialize the simulations.

# Step 1: Set Up the Colab Environment

# 1. Remove existing file to prevent duplicate versions
!rm -f wk_profiles.py

# First, download the wk_profiles.py file directly into Colab session and install the necessary data science libraries.

# 1. Download the raw script directly from GitHub

# ==============================================================================
# INITIALIZATION: Environment Setup & Meteorological Imports
# ==============================================================================
import os
import sys
import importlib.util

# User-requested imports for thermodynamic analysis
import numpy as np
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units
from metpy.plots import SkewT

print("--- Starting Environment Setup ---")

# 1. Clean up old versions of the script to prevent naming conflicts
if os.path.exists('wk_profiles.py'):
    !rm -f wk_profiles.py
    print("✓ Removed existing wk_profiles.py")

# 2. Download the raw wk_profiles.py script used in MWR-D-23-0262.1
!wget -q https://raw.githubusercontent.com/tsupinie/wk-profiles/main/wk_profiles.py
print("✓ Downloaded fresh copy of wk_profiles.py from tsupinie/wk-profiles")

# 3. Conditionally install dependencies only if they are missing
required_pkgs = ["numpy", "matplotlib", "metpy"]
for pkg in required_pkgs:
    if importlib.util.find_spec(pkg) is None:
        print(f"Installing {pkg}...")
        !{sys.executable} -m pip install -q {pkg}
    else:
        print(f"✓ {pkg} dependency is already satisfied")

# 4. Import the specific Weisman-Klemp profile functions
try:
    from wk_profiles import wk_sounding, find_wk_params
    print("\n[SUCCESS] All libraries imported. Ready to generate modulated profiles.")
except ImportError:
    print("\n[ERROR] Setup failed. wk_profiles.py could not be loaded.")

# Step 2: Define and Modulate the Profile

# Define the heights you want to calculate (e.g., 0 to 20km every 100m)
heights = np.arange(0, 20001, 100)

# Example: Generating two different profiles to see "Modulation"
# We will keep CAPE constant at 2000 J/kg but change the LCL height.
configs = [
    {"sbcape": 2000, "sblcl": 500,  "label": "Low LCL (Moist)"},
    {"sbcape": 2000, "sblcl": 1500, "label": "High LCL (Drier)"}
]

results = []

for cfg in configs:
    # 1. Find the exact mathematical parameters for this configuration
    params = find_wk_params(tsfc=293.0, psfc=101325.0, 
                            sbcape=cfg['sbcape'], sblcl=cfg['sblcl'])
    
    # 2. Create the sounding generator object
    prof_gen = wk_sounding(**params)
    
    # 3. Generate the data: Temperature (K), Dewpoint (K), and Pressure (Pa)
    temp, dewp, pres = prof_gen(heights)
    
    results.append({'temp': temp, 'dewp': dewp, 'pres': pres, 'label': cfg['label']})

# Step 3: Visualize the Result

plt.figure(figsize=(10, 8))

for res in results:
    # Plot Temperature
    plt.plot(res['temp'] - 273.15, heights / 1000, label=f"Temp - {res['label']}")
    # Plot Dewpoint (shows the moisture modulation)
    plt.plot(res['dewp'] - 273.15, heights / 1000, '--', label=f"Dewp - {res['label']}")

plt.xlabel("Temperature (C)")
plt.ylabel("Height (km)")
plt.title("Modulated Weisman-Klemp Profiles")
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Export for CM1 Simulations

# 1. The Full Implementation in Colab

# This code calculates the derived meteorological variables and creates a two-panel plot: one for the vertical profiles used by the model and one for the standard Skew-T Log-P diagram
# used by meteorologists.

# Documentation for calculations in metpy:
# https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.html


def plot_cm1_input(z, p, t, td, label="Modulated Profile"):
    # 1. Convert inputs to MetPy units for accurate calculation
    p_pts = p * units.pascal
    t_pts = t * units.kelvin
    td_pts = td * units.kelvin

    # 2. Calculate derived variables for CM1
    theta = mpcalc.potential_temperature(p_pts, t_pts)
    
    # Calculate saturation vapor pressure at dewpoint temperature
    vapor_pressure = mpcalc.saturation_vapor_pressure(td_pts)
    # Pass partial pressure and total pressure as positional arguments to mixing_ratio
    mixing_ratio = mpcalc.mixing_ratio(vapor_pressure, p_pts)

    # 3. Create the Visualization
    fig = plt.figure(figsize=(12, 6))

    # Panel A: Vertical Profile (What CM1 reads)
    ax1 = fig.add_subplot(121)
    ax1.plot(theta.magnitude, z/1000, 'r', label='Potential Temp (theta)')
    ax1.set_xlabel('Potential Temperature (K)', color='r')
    ax1.set_ylabel('Height (km)')

    ax1b = ax1.twiny() # Create second x-axis for mixing ratio
    ax1b.plot(mixing_ratio.magnitude * 1000, z/1000, 'g', label='Mixing Ratio (qv)')
    ax1b.set_xlabel('Mixing Ratio (g/kg)', color='g')
    ax1.set_title(f"CM1 Input Variables: {label}")
    ax1.grid(True)

    # Panel B: Skew-T Log-P (Meteorological View)
    skew = SkewT(fig, rotation=45, subplot=122)
    skew.plot(p_pts, t_pts.to('degC'), 'r', linewidth=2)
    skew.plot(p_pts, td_pts.to('degC'), 'g', linewidth=2)

    # Add parcel path (CAPE)
    prof = mpcalc.parcel_profile(p_pts, t_pts[0], td_pts[0]).to('degC')
    skew.plot(p_pts, prof, 'k--', alpha=0.7, label='Parcel Path')
    skew.shade_cape(p_pts, t_pts.to('degC'), prof)

    skew.ax.set_title("Thermodynamic Environment")
    plt.tight_layout()
    plt.show()

# --- EXECUTION ---
# Using the data from the previous step
for res in results:
    plot_cm1_input(heights, res['pres'], res['temp'], res['dewp'], label=res['label'])

# Top Plot: This plot visualizes the thermodynamic vertical structure of the atmosphere generated by the wk-profiles package. 
# It shows how changing the surface moisture "modulates" the environment where a simulated storm will grow.
# 12 km where lines become vertical is beginning of stratosphere.

# Simulates two worlds: a drier one with a higher cloud base and a more humid one with a lower cloud base.

# Panel A (Left): This panel shows the vertical profiles of potential temperature (red line) and mixing ratio (green line) as CM1 simulation would interpret them. 
# You can see how these variables change with height (in kilometers) for each of your defined profiles.

# Panel B (Right): This is a Skew-T Log-P diagram, a standard meteorological chart. It displays the temperature (red line) and dewpoint temperature (green line) profiles. 
# The black dashed line indicates the path of a parcel lifted from the surface, and the shaded area represents the Convective Available Potential Energy (CAPE).

# These plots effectively illustrate the differences between your 'Low LCL (Moist)' and 'High LCL (Drier)' profiles, 
# especially how the mixing ratio and thermodynamic environment are modulated by the LCL height while keeping CAPE constant.
