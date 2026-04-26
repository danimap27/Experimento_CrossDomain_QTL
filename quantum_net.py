import pennylane as qml
import torch
import torch.nn as nn

class HybridQuantumNet(nn.Module):
    """
    Hybrid Classical-Quantum Neural Network encapsulating a PyTorch neural network
    with a PennyLane Variational Quantum Circuit (VQC).
    """
    def __init__(self, ansatz='A', n_qubits=4, n_layers=2):
        super(HybridQuantumNet, self).__init__()
        self.n_qubits = n_qubits
        self.ansatz = ansatz
        self.n_layers = n_layers
        
        # Define the quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Determine weight shapes based on the ansatz
        if ansatz == 'A':
            # Strongly Entangling Layers
            self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        elif ansatz == 'B':
            # Basic Entangler Layers
            self.weight_shapes = {"weights": (n_layers, n_qubits)}
        elif ansatz == 'C':
            # Tree Tensor Network (TTN) - Custom hierarchical structure
            # Let's parameterize each block with 2 angles (Ry, Rz)
            n_blocks = n_qubits - 1
            self.weight_shapes = {"weights": (n_layers, n_blocks, 2)}
        else:
            raise ValueError("Ansatz must be 'A', 'B', or 'C'.")

        self.vqc = qml.qnn.TorchLayer(self._qnode(), self.weight_shapes)
        # Add a classical layer to scale the [-1, 1] expectation values into proper logits
        self.fc = nn.Linear(2, 2)

    def _qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # 1. Classical-to-Quantum Encoding
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            
            # 2. Parameterized Topological Ansatz
            if self.ansatz == 'A':
                qml.StronglyEntanglingLayers(weights=weights, wires=range(self.n_qubits))
            
            elif self.ansatz == 'B':
                qml.BasicEntanglerLayers(weights=weights, wires=range(self.n_qubits))
            
            elif self.ansatz == 'C':
                # Custom TTN Architecture
                num_layers = weights.shape[0]
                for l in range(num_layers):
                    block_idx = 0
                    # Layer 1 of TTN (pairs)
                    for i in range(0, self.n_qubits, 2):
                        qml.RY(weights[l, block_idx, 0], wires=i)
                        qml.RZ(weights[l, block_idx, 1], wires=i+1)
                        qml.CNOT(wires=[i, i+1])
                        block_idx += 1
                    
                    # Layer 2 of TTN (merging)
                    step = 2
                    for i in range(0, self.n_qubits, step*2):
                        qml.RY(weights[l, block_idx, 0], wires=i+1)
                        qml.RZ(weights[l, block_idx, 1], wires=i+step+1)
                        qml.CNOT(wires=[i+1, i+step+1])
                        block_idx += 1
            
            # 3. Measurement (Expectation value for 2 output classes)
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]
            
        return circuit

    def forward(self, x):
        # Forward pass through the VQC
        out = self.vqc(x)
        # Post-processing layer for CrossEntropy compatibility
        out = self.fc(out)
        return out
