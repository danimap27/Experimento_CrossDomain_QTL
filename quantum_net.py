import pennylane as qml
import torch
import torch.nn as nn

# IBM Heron r2 reference noise values (median calibration ranges, e.g. ibm_torino, ibm_marrakesh)
HERON_R2_NOISE = {
    "p_1q": 3.0e-4,    # 1-qubit gate depolarizing error
    "p_2q": 3.0e-3,    # 2-qubit gate depolarizing error
    "p_readout": 1.5e-2,  # readout / measurement error
}


class HybridQuantumNet(nn.Module):
    """
    Hybrid Classical-Quantum Neural Network encapsulating a PyTorch neural network
    with a PennyLane Variational Quantum Circuit (VQC).

    If `noise=True`, the circuit is executed on the `default.mixed` device with
    depolarizing channels modelling IBM Heron r2 calibration data.
    """

    def __init__(self, ansatz='A', n_qubits=4, n_layers=2, noise=False, noise_params=None):
        super(HybridQuantumNet, self).__init__()
        self.n_qubits = n_qubits
        self.ansatz = ansatz
        self.n_layers = n_layers
        self.noise = noise
        self.noise_params = noise_params if noise_params is not None else HERON_R2_NOISE

        # Mixed-state simulator required for depolarizing channels
        if self.noise:
            self.dev = qml.device("default.mixed", wires=n_qubits)
        else:
            self.dev = qml.device("default.qubit", wires=n_qubits)

        if ansatz == 'A':
            self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        elif ansatz == 'B':
            self.weight_shapes = {"weights": (n_layers, n_qubits)}
        elif ansatz == 'C':
            n_blocks = n_qubits - 1
            self.weight_shapes = {"weights": (n_layers, n_blocks, 2)}
        else:
            raise ValueError("Ansatz must be 'A', 'B', or 'C'.")

        self.vqc = qml.qnn.TorchLayer(self._qnode(), self.weight_shapes)
        self.fc = nn.Linear(2, 2)

    # ------------------------------------------------------------------
    # Noise helpers
    # ------------------------------------------------------------------
    def _noise_1q(self, wire):
        if self.noise:
            qml.DepolarizingChannel(self.noise_params["p_1q"], wires=wire)

    def _noise_2q(self, wires):
        if self.noise:
            for w in wires:
                qml.DepolarizingChannel(self.noise_params["p_2q"], wires=w)

    def _noise_readout(self):
        """Bit-flip on each measured wire to emulate readout error."""
        if self.noise:
            for w in range(self.n_qubits):
                qml.BitFlip(self.noise_params["p_readout"], wires=w)

    def _strongly_entangling_noisy(self, weights):
        """Manual StronglyEntanglingLayers with depolarizing channels."""
        n_layers = weights.shape[0]
        n_wires = self.n_qubits
        for l in range(n_layers):
            for q in range(n_wires):
                qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
                self._noise_1q(q)
            r = (l % (n_wires - 1)) + 1
            for q in range(n_wires):
                target = (q + r) % n_wires
                qml.CNOT(wires=[q, target])
                self._noise_2q([q, target])

    def _basic_entangler_noisy(self, weights):
        n_layers = weights.shape[0]
        n_wires = self.n_qubits
        for l in range(n_layers):
            for q in range(n_wires):
                qml.RX(weights[l, q], wires=q)
                self._noise_1q(q)
            for q in range(n_wires):
                target = (q + 1) % n_wires
                qml.CNOT(wires=[q, target])
                self._noise_2q([q, target])

    # ------------------------------------------------------------------
    def _qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # 1. Encoding
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            if self.noise:
                for w in range(self.n_qubits):
                    self._noise_1q(w)

            # 2. Ansatz
            if self.ansatz == 'A':
                if self.noise:
                    self._strongly_entangling_noisy(weights)
                else:
                    qml.StronglyEntanglingLayers(weights=weights, wires=range(self.n_qubits))

            elif self.ansatz == 'B':
                if self.noise:
                    self._basic_entangler_noisy(weights)
                else:
                    qml.BasicEntanglerLayers(weights=weights, wires=range(self.n_qubits))

            elif self.ansatz == 'C':
                num_layers = weights.shape[0]
                for l in range(num_layers):
                    block_idx = 0
                    for i in range(0, self.n_qubits, 2):
                        qml.RY(weights[l, block_idx, 0], wires=i)
                        self._noise_1q(i)
                        qml.RZ(weights[l, block_idx, 1], wires=i + 1)
                        self._noise_1q(i + 1)
                        qml.CNOT(wires=[i, i + 1])
                        self._noise_2q([i, i + 1])
                        block_idx += 1

                    step = 2
                    for i in range(0, self.n_qubits, step * 2):
                        qml.RY(weights[l, block_idx, 0], wires=i + 1)
                        self._noise_1q(i + 1)
                        qml.RZ(weights[l, block_idx, 1], wires=i + step + 1)
                        self._noise_1q(i + step + 1)
                        qml.CNOT(wires=[i + 1, i + step + 1])
                        self._noise_2q([i + 1, i + step + 1])
                        block_idx += 1

            # 3. Readout error + measurement
            self._noise_readout()
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]

        return circuit

    def forward(self, x):
        out = self.vqc(x)
        out = self.fc(out)
        return out
