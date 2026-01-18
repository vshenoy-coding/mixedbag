# Adopted from instructions in https://github.com/Qiskit/qiskit

# First check if qiskit is installed

import importlib.util
import sys
import subprocess

def prepare_qiskit():
    """Checks for Qiskit and installs it along with visualization dependencies."""
    # Qiskit 1.0+ uses 'qiskit' as the namespace
    packages = ["qiskit", "qiskit-aer", "pylatexenc", "matplotlib"]
    
    for pkg in packages:
        # Check if the package is already available
        if importlib.util.find_spec(pkg.replace('-', '_')) is None:
            print(f"--- {pkg} not found. Installing... ---")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        else:
            print(f"✓ {pkg} is ready.")

prepare_qiskit()

# Final verification and version check
import qiskit
print(f"\nQiskit setup complete. Version: {qiskit.__version__}")

# Create example quantum circuit using the QuantumCircuit class

import numpy as np
from qiskit import QuantumCircuit

# 1. A quantum circuit for preparing the quantum state |000> + i |iii / √2
qc = QuantumCircuit(3)
qc.h(0)                # generate superposition
qc.p(np.pi / 2, 0)     # add quantum phase
qc.cx(0, 1)            # 0th-qubit-Controlled-NOT gate on 1st qubit
qc.cx(0, 2)            # 0th-qubit-Controlled-NOT gate on 2nd qubit

# Creates an entangled state known as a GHZ state ( | 000 ⟩ + i | 111 ⟩ ) / 2 . 
# Uses the standard quantum gates: Hadamard gate (h), Phase gate (p), and CNOT gate (cx).

# Now that the first quantum circuit has been made, choose which primitive to use.

# Starting with the Sampler primitive, use measure_all(inplace=False) to get a copy
# of the circuit in which all the qubits are measured

# 2. Add the classical output in the form of measurement of all qubits
qc_measured = qc.measure_all(inplace=False)

# 3. Execute using the Sampler primitive
from qiskit.primitives import StatevectorSampler
sampler = StatevectorSampler()
job = sampler.run([qc_measured], shots=1000)
result = job.result()
print(f" > Counts: {result[0].data['meas'].get_counts()}")

# 000 50% of the time and 111 50% of the time up to statistical fluctuations

# Use the quantum information toolbox to create operator
# XXY + XYX + YXX - YYY and pass it to the run() function,
# along with your quantum circuit.

# Use qc circuit created earlier since Estimator requires 
# circuit without measurements.

# 4. Define the observable to be measured
from qiskit.quantum_info import SparsePauliOp
operator = SparsePauliOp.from_list([("XXY", 1), ("XYX", 1), ("YXX", 1), ("YYY", -1)])

# 5. Execute using the Estimator primitive
from qiskit.primitives import StatevectorEstimator
estimator = StatevectorEstimator()
job = estimator.run([(qc, operator)], precision = 1e-3)
result = job.result()
print(f" > Expectation values: {result[0].data.evs}")

# Gives an outcome of ~4
# Assigning a value of +/-1 to each single-qubit operator X and Y does not allow
# you to achieve this outcome:


import itertools

# 1. Define the possible values for local hidden variables
values = [1, -1]

# 2. Create all possible combinations for (X1, Y1, X2, Y2, X3, Y3)
# There are 2^6 = 64 possible classical states
combinations = list(itertools.product(values, repeat=6))

max_classical_value = -float('inf')
best_combo = None

for c in combinations:
    x1, y1, x2, y2, x3, y3 = c
    
    # Calculate each term of the observable: XXY + XYX + YXX - YYY
    term1 = x1 * x2 * y3
    term2 = x1 * y2 * x3
    term3 = y1 * x2 * x3
    term4 = y1 * y2 * y3
    
    total = term1 + term2 + term3 - term4
    
    if total > max_classical_value:
        max_classical_value = total
        best_combo = c

print("--- Classical Locality Check ---")
print(f"Max classical outcome possible: {max_classical_value}")
print(f"Example variables (x1,y1,x2,y2,x3,y3) for max value: {best_combo}")

# 3. Reference your Quantum result
print("\n--- Quantum Result ---")
print("Expected Quantum value (GHZ State): 4.0")
print(f"Violation Magnitude: {4.0 / max_classical_value}x higher than classical limit")

# This is a version of Bell's Theorem. 
# It proves that no "common sense" classical theory 
# (where objects have defined properties before you look at them) can explain the correlations 
# measured by Qiskit. 

# The fact that your Estimator returned a value near 4 is direct evidence of 
# Quantum Non-locality 
# (measurement results of two or more particles are correlated in ways 
# that cannot be explained by classical physics, even when those particles are separated 
# by vast distances).

# Running a quantum circuit on hardware requires rewriting to the basis gates and 
# connectivity of the quantum hardware. 

# The tool that does this is the transpiler.

# Qiskit includes transpiler passes for synthesis, optimization, mapping, and scheduling.

# However, it also includes a default compiler, which works very well in most examples. 

# The following code will map the example circuit to the basis_gates = 
# ["cz", "sx", "rz"] and a bidirectional linear chain of qubits 0 ↔ 1 ↔ 2
# with the coupling_map = [[0, 1], [1, 0], [1, 2], [2, 1]]

from qiskit import transpile
from qiskit.transpiler import Target, CouplingMap
target = Target.from_configuration(
    basis_gates=["cz", "sx", "rz"],
    coupling_map=CouplingMap.from_line(3),
)
qc_transpiled = transpile(qc, target=target)
print(qc_transpiled)

# global phase pi/2
