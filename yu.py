import numpy as np
import qutip as qt
from qiskit.extensions import UnitaryGate
from qiskit import QuantumCircuit, Aer
from qiskit.visualization import plot_bloch_multivector
from itertools import product
from qiskit.providers.aer import StatevectorSimulator
def csh_gate(qc, qubit, matrix):
    custom_gate = UnitaryGate(matrix, label="CSH")
    qc.append(custom_gate, [qubit])

def custom_hadamard_matrix(x, nd):
    factor = 1/np.sqrt(x**(2*nd) - nd)
    return [[factor, factor], [factor, -factor]]

# Given values:
x = 2.5000000000000004  # Explicitly set value for x
nd = 0.5  # Sample value for nd

matrix = custom_hadamard_matrix(x, nd)
CSH = custom_hadamard_matrix(x, nd)
def apply_gate(state, gate_name):
    gates = {
        "I": qt.qeye(2),   # Identity
        "X": qt.sigmax(),  # Pauli-X
        "Y": qt.sigmay(),  # Pauli-Y
        "Z": qt.sigmaz(),  # Pauli-Z
        "H": qt.hadamard_transform(),
        "CSH": CSH
        # Add other gates as required
    }
    gate = gates.get(gate_name)
    if gate:
        return gate * state
    else:
        raise ValueError(f"Gate {gate_name} not recognized.")

def apply_gate(qc, gate_name):
    if gate_name == "I":
        pass  # Identity, do nothing
    elif gate_name == "X":
        qc.x(0)
    elif gate_name == "Y":
        qc.y(0)
    elif gate_name == "Z":
        qc.z(0)
    elif gate_name == "H":
        qc.h(0)
    elif gate_name == "S":
        qc.s(0)
    elif gate_name == "T":
        qc.t(0)
    elif gate_name == "SDG":
        qc.sdg(0)
    elif gate_name == "TDG":
        qc.tdg(0)
    elif gate_name == "CSH":
        csh_gate(qc, 0, matrix)
def plot_gate_on_bloch(gate_sequence):
    qc = QuantumCircuit(1)

    for gate in gate_sequence:
        apply_gate(qc, gate)

    simulator = Aer.get_backend('statevector_simulator')
    result = simulator.run(qc).result()
    statevector = result.get_statevector(qc)

    fig = plot_bloch_multivector(statevector)
    filename = f'bloch_{"_".join(gate_sequence)}.png'
    fig.savefig(filename)
    print(f"Saved {filename}")

# List of single qubit gates
gates = ["I", "X", "Y", "Z", "H", "S", "T", "SDG", "TDG", "CSH"]

# Generating all combinations and saving their visualizations
for r in range(1, len(gates) + 1):
    for subset in product(gates, repeat=r):
        plot_gate_on_bloch(subset)
def parse_quantum_output(output_string):
    # Split the string by lines
    lines = output_string.strip().split("\n")

    results = []
    for line in lines:
        # Split each line by commas
        parts = line.split(",")

        # The first item is the count, the rest is the quantum state
        count = int(parts[0])
        state = parts[1:]

        # Create a dictionary with column headers and values
        result_dict = {"Count": count}
        for i, qubit in enumerate(state):
            # Check if the qubit is '...', if so, continue to the next iteration
            if qubit == '...':
                continue
            result_dict[str(i)] = int(qubit)

        results.append(result_dict)

    return results

# Convert counts to DataFrame
df = pd.DataFrame(list(counts.items()), columns=['Eigenstate', 'Count'])

# Break down the 192-qubit eigenstate to 192 columns
df_eigenstate = df['Eigenstate'].apply(lambda x: pd.Series(list(x)))
df = pd.concat([df, df_eigenstate], axis=1)
df.drop(columns=['Eigenstate'], inplace=True)

# If you want to truncate or limit data to 32000 instances
df = df.head(32000)

# Save the DataFrame to a CSV file
df.to_csv('results.csv', index=False)
b = qt.Bloch()
b.add_states(state)
b.show()
