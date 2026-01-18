# Step 1. The Pythonic Dependency Guard

import importlib.util
import sys
import subprocess

def smart_install(packages):
    """Install packages only if they are not already available."""
    to_install = [pkg for pkg in packages if importlib.util.find_spec(pkg.split('-')[0]) is None]

    if not to_install:
        print("All dependencies satisfied.")
    else:
        print(f"Installing missing packages: {', '.join(to_install)}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])

# Define our stack
smart_install(["qiskit", "qiskit-aer", "matplotlib", "pylatexenc"])

# Import after ensuring packages exist
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Step 2: Modular circuit Construction 
# Generate GHZ state and apply basis rotation X, Y, or Z) without rewriting code.

def get_ghz_parity_circuit(basis="XXY"):
    """
    Factory to create a GHZ state |000> + i|111>
    rotated into a specific measurement basis.
    """
    num_qubits = len(basis)
    qc = QuantumCircuit(num_qubits)

    # 1. Entanglement Layer
    qc.h(0)
    qc.p(np.pi / 2, 0) # The imaginary phase for Mermin violation
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    qc.barrier()

    # 2. Basis Rotation Layer (Pythonic mapping)
    for i, b in enumerate(basis.upper()):
        if b == 'X':
            qc.h(i)
        elif b == 'Y':
            qc.sdg(i)
            qc.h(i)
        # Z basis is the default, no rotation needed

    qc.measure_all()
    return qc

# Step 3: Data Processing & Parity Mapping

# Use dictionary comprehension to map raw "counts" from the simulator into
# a structured format containing probability and parity (Even/Odd)

def process_counts(counts):
    """Map binary outcomes to parity categories."""
    total = sum(counts.values())

    # Parity logic: (-1)^sum(bits). 0 is Even (+1), 1 is Odd (-1)
    return {
        bit: {
            'prob': c / total,
            'parity': 'Even' if sum(int(b) for b in bit) % 2 == 0 else 'Odd'
        }
        for bit, c in counts.items()
    }

# Generating the Parity Plot

# Use matplotlib to create the visualization and color-code based on parity calculated
# in Step 3. to make Quantum Correlation visually apparent

def create_parity_plot(basis_string="XXY"):
    # Execute
    qc = get_ghz_parity_circuit(basis_string)
    backend = AerSimulator()
    counts = backend.run(qc, shots=1024).result().get_counts()
    data = process_counts(counts)

    # Sort for consistent X-axis
    sorted_bits = sorted(data.keys())
    probs = [data[b]['prob'] for b in sorted_bits]
    colors = ['#2ecc71' if data[b]['parity'] == 'Even' else '#e74c3c' for b in sorted_bits]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.bar(sorted_bits, probs, color=colors)
    plt.title(f"Parity Correlation for {basis_string} Basis")
    plt.ylabel("Probability")
    plt.xlabel("Outcome")

    # Pythonic Legend
    from matplotlib.lines import Line2D
    plt.legend([Line2D([0], [0], color='#2ecc71', lw=4), Line2D([0], [0], color='#e74c3c', lw=4)],
               ['Even Parity', 'Odd Parity'])
    plt.show()

# Run the final result
create_parity_plot("XXY")


# To show the transition from an ideal simulator a "noisy" one representing a real quantum
# computer, introduce a Noise Model. On a real quantum computer, environmental interference
# and gate imperfections cause decoherence, which leads to the appearance of "missing" parity bits
# (red bars) previously suppressed by quantum interference.

# 1. The Pythonic "Noisy" Setup
# Use AerSimulator along with built-in NoiseModel to simulate basic gate errors.
# This demonstrates why real hardware rarely shows "perfect" 100% correlation.

import importlib.util

# Pythonic dependency guard for noise simulation
if importlib.util.find_spec("qiskit_aer") is None:
    print("Installing qiskit-aer for noise simulation...")
    import subprocess, sys
    suprocess.check_call([sys.executable, "-m", "pip", "install", "qiskit-aer"])

from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator

def get_noisy_backend(error_rate=0.05):
    """Creates a simulator with a simple depolarizing noise model."""
    noise_model = NoiseModel()
    # Apply error to q-qubit gates (h, p) and 2-qubit gates (cx)
    error_1 = depolarizing_error(error_rate, 1)
    error_2 = depolarizing_error(error_rate * 2, 2)

    noise_model.add_all_qubit_quantum_error(error_1, ['h', 'p'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    return AerSimulator(noise_model=noise_model)

# 2. Executing the Comparison

# Now compare the ideal results earlier with a noisy run. This is where the red bars will finally appear.

# 1. Generate the same XXY circuit from previous step
qc_noisy = build_parity_circuit("XXY")

# 2. Run on the noisy simulator
noisy_backend = get_noisy_backend(error_rate=0.03) # 3% gate error
noisy_counts = noisy_backend.run(qc_noisy, shots=2048).result().get_counts()

# 3. Process the data 
noisy_data = process_counts(noisy_counts)

# 3. Visualizing the "Bleed" into Odd Parity

# Green bars (Even Parity) have lowered in height, and the red bars (Odd Parity)
# have emerged from zero.

def plot_noisy_parity(data, basis_name):
    bits = sorted(data.keys())
    probs = [data[b]['prob'] for b in bits]
    # Pythonic color assignment: Green for Even, Red for Odd
    colors = ['#2ecc71' if data[b]['parity] == 'Even' else '#e74c3c' for b in bits]

    plt.figure(figsize=(12, 6))
    plt.bar(bits, probs, color=colors)

    plt.title(f"Noisy Parity Plot (Real Device Simulation): {basis_name}")
    plt.ylabel("Probability")
    plt.xlabel("Outcome")

    # Add labels to the 'Red' bars to show they are no longer zero
    for i, p in enumerate(probs):
        if p > 0.01: # Only label visible bars
            plt.text(i, p + 0.01, f{p:.2f}", ha='center')

    plt.ylim(0, 0.4)
    plt.show()

plot_noisy_parity(noisy_data, "XXY")
