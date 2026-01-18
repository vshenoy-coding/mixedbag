# mixedbag
A mixed bag of coding projects to work on with no unifying theme.

**Quick Start: Running a New Experiment**

To run a new sensitivity experiment similar to those in **Brown and Marion (2024)**, you primarily need to modify the configs list in the provided Python script. This allows you to "dial in" the thermodynamic environment before exporting it to CM1.

**1. Key Variables for Modulation**

To create a suite of different simulations, adjust the following parameters within the find_wk_params() function call:

- sbcape: Controls the total energy (fuel) available for the storm. In the study, researchers often keep this constant while varying other factors to isolate specific effects.

- sblcl: Controls the moisture levels through the Lifting Condensation Level. A lower SBLCL (e.g., 500m) represents a moister world, while a higher SBLCL (e.g., 1500m) represents a drier world.

- tsfc: Set the surface temperature (in Kelvin). Standard Weisman-Klemp profiles typically start around 293.0 K (20°C).

- psfc: Set the surface pressure (in Pascals). Standard value is 101325.0 Pa.

**2. Modifying the Configuration Loop**

In your code, you can define multiple "worlds" to test at once:

# Edit this list to define your experimental members
```python
configs = [
    {"sbcape": 2000, "sblcl": 750,  "label": "Experiment_A_Moist"},
    {"sbcape": 2000, "sblcl": 1250, "label": "Experiment_B_Intermediate"},
    {"sbcape": 2000, "sblcl": 1750, "label": "Experiment_C_Dry"}
]
```

**3. Exporting to CM1**

After running the script, the save_cm1_sounding function will generate a text file for each configuration.

1. Locate the generated file in the Colab sidebar (e.g., input_sounding_low_lcl).

2. Rename this file to input_sounding for use in your CM1 run directory.

3. Ensure your namelist.input in CM1 is set to read from an external sounding file (isnd = 3).

**Why These Variables?**

The authors of the paper utilized these specific modulations because the LCL (Lifting Condensation Level) acts as a proxy for low-level moisture. By keeping CAPE constant and only moving the LCL, they could determine if a storm's rotation was changed by the moisture itself or the energy of the environment.

## 🤖 AI Disclosure & Verification
This project utilized **Google Gemini** for code generation and logic assistance.

To ensure reliability, all generated code has been:
*   **Verified:** Manually reviewed for security and logic.
*   **Tested:** Executed in **Google Colab** environments to ensure error-free performance.

*   All AI-suggested fixes were manually verified through trace analysis in **Google Gemini** and successfully validated in **Google Colab**.

---
*Co-authored-by: Google Gemini <gemini.google.com>*
*Tested-in: Google Colab*
