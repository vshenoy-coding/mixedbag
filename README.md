# mixedbag
A mixed bag of coding projects to work on with no unifying theme.

## qnk_vqe_burgers.py

This repository implements a novel pipeline that maps atmospheric thermodynamic environments to Variational Quantum Eigensolver (VQE) states to predict storm morphology. By solving a 1D Burgers' Equation on a quantum backend, we identify "Quantum Signatures" that correspond to specific weather families.

🌪️ The "Quantum-to-Storm" Pipeline

Regime Mapping: Categorizes atmospheric soundings by SBCAPE and SBLCL into three distinct families: Moist, Intermediate, and Dry.

Quantum Fluid Simulation: Solves the Burgers' Equation using a 3-qubit VQE with the EstimatorV2 primitive.

Morphology Classification: Uses Quantum Residual Energy and KL Divergence to predict if a storm will maintain a stable squall line or fragment into discretized cells.

📊 Core Findings
1. The Fluid-Atmosphere Coupling

A direct correlation was discovered between SBLCL heights and the stability of the VQE solution:

Moist Regime (SBLCL < 1125m): Produces a smooth, single-peak velocity profile with high symmetry (Even Parity) and low residual energy (≈−2.15).

Dry Regime (SBLCL > 1375m): Triggers a "turbulent" bi-modal velocity profile with high residual energy (≈−0.75), signaling a transition to discretized cell morphology.

2. Turbulence Metrics
|Metric|Moist Regime|Dry Regime|Scientific Significance|
| ---  | --- | --- | --- |
|Residual Energy|−2.15|−0.75|"Higher energy indicates a ""stiff"" fluid flow."|

|TV Distance|0.00 (Base)|0.5000| 50% of the fluid mass shifts spatially in dry air.|

|KL Divergence|0.00 (Base)|0.6960| Dry regimes add ~1 bit of informational complexity.|

## qml_cluster_analysis.py

Implements an unsupervised learning layer over the Quantum Similarity Matrix. It uses Agglomerative Hierarchical Clustering and calculates the Silhouette Score to determine the optimal number of atmospheric regimes. This automates the process of identifying "tipping points" in storm morphology, reducing the need for manual sensitivity tests in the CM1 model.

## quantum_circuit_parity_plot.py

Recreated foundational proofs of quantum non-locality and provide a robust diagnostic framework for quantum state validation.

**Quantum Information Implementation:**

**Parity Visualization:** Developed a "Pythonic" workflow to generate parity plots for the complex GHZ state (2​∣000⟩+i∣111⟩​), demonstrating perfect even-parity correlation in the XXY basis.

**Mermin Inequality Proof:** Executed a classical brute-force simulation using itertools to prove that the maximum local hidden variable outcome is 2.0, sharply contrasted against a measured quantum expectation value of 4.0.

**Noise Modeling:** Introduced a depolarizing noise model to simulate decoherence and gate errors inherent in real quantum hardware, visually demonstrating the "leakage" or emergence of odd-parity (red) results that are theoretically suppressed in ideal systems.

**Fidelity Analysis:** Implemented a post-execution diagnostic using Hellinger Fidelity to mathematically quantify the overlap between ideal probability distributions and noisy hardware results, providing a single metric to evaluate state preparation success.

**Hardware Transpilation:** Utilized the Qiskit transpiler to map idealized circuits to a hardware-specific target (linear 0-1-2 coupling map) using a restricted cz, sx, and rz basis gate set.

## qiskit_first_example_circuit.py

**Quantum Non-Locality & Transpilation Proofs**

**Core Objectives:**
1. Prepare GHZ state with specific relative phase (π/2).
2. Extract expectation values using Qiskit Primtives (Estimator).
3. Prove non-classicality via Mermin Inequality (Quantum 4 vs Classical 2).
4. Transpile for a 0-1-2 linear coupling map with basis gates sx, rz, cz.

Hardware Mapping:

Target: Linear Chain (0 ↔ 1 ↔ 2)

Global Phase Result: π/2

## Brown_Marion_Coniglio_2024_Figure1.py

Recreation of first figure showing analytic Weisman-Klemp thermodynamic profile and the specific wind profiles modulated by the low-level shear orientation angle α.

## wk_cm1_generator.py

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
In your script, you can define multiple "worlds" to test at once by editing the `configs` list. This is the primary way to replicate the sensitivity tests performed in the study.

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

The authors of the paper utilized these specific modulations because the LCL (Lifting Condensation Level) acts as a proxy for low-level moisture. 
By keeping CAPE constant and only moving the LCL, they could determine if a storm's rotation was changed by the moisture itself or the energy of the environment.

## 🤖 AI Disclosure & Verification
This project utilized **Google Gemini** for code generation and logic assistance.

To ensure reliability, all generated code has been:
*   **Verified:** Manually reviewed for security and logic.
*   **Tested:** Executed in **Google Colab** environments to ensure error-free performance.

*   All AI-suggested fixes were manually verified through trace analysis in **Google Gemini** and successfully validated in **Google Colab**.


*Tested-in: Google Colab*
