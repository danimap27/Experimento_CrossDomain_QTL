import math

import pennylane as qml
import torch
import torch.nn as nn

# IBM Heron r2 reference noise values (median calibration ranges, e.g. ibm_torino, ibm_marrakesh)
HERON_R2_NOISE = {
    "p_1q": 3.0e-4,    # 1-qubit gate depolarizing error
    "p_2q": 3.0e-3,    # 2-qubit gate depolarizing error
    "p_readout": 1.5e-2,  # readout / measurement error
}

# Older / lower-fidelity NISQ reference (representative of pre-Heron generation
# IBM Eagle / Falcon devices: ~10x worse 2Q errors).
LEGACY_NISQ_NOISE = {
    "p_1q": 1.0e-3,
    "p_2q": 1.5e-2,
    "p_readout": 3.0e-2,
}

NOISE_PROFILES = {
    "ideal":       None,
    "heron_r2":    HERON_R2_NOISE,
    "legacy_nisq": LEGACY_NISQ_NOISE,
}


def get_noise_profile(name: str):
    """Return (noise_enabled, noise_params) for a profile name."""
    if name not in NOISE_PROFILES:
        raise ValueError(f"Unknown noise profile {name!r}. "
                         f"Available: {list(NOISE_PROFILES)}")
    params = NOISE_PROFILES[name]
    return (params is not None), params


def ttn_pairs(n_qubits: int):
    """Merge schedule for the TTN ansatz (ansatz 'C').

    Returns the list of (control, target) wire pairs of the binary-tree
    contraction, ordered bottom-up: level 0 couples (0,1), (2,3), ...; level 1
    couples (1,3), (5,7), ...; and so on until the root pair. For n_qubits=4
    this reproduces exactly the historical schedule [(0,1), (2,3), (1,3)]
    (3 = n_qubits - 1 blocks). For any n_qubits >= 2 the schedule is
    crash-free (the previous implementation crashed for n_qubits != 4); for
    powers of two it has exactly n_qubits - 1 blocks, for other sizes the tree
    is partial (still valid wiring, just fewer merges).

    NOTE (audit A7 / finding H9): with the default readout on wires (0, 1)
    and the CNOT chain propagating towards the root, the parameters of the
    branches that never reach the readout wires have exactly zero gradient
    ("readout-limited" variant). This variant is kept as the default for
    comparability with the submitted runs; the `ttn_readout='root'` variant
    measures the root pair instead and must be reported as an effective-
    parameter ablation if used.
    """
    pairs = []
    level = 0
    while (1 << level) < n_qubits:
        stride = 1 << (level + 1)
        half = 1 << level
        for base in range(0, n_qubits, stride):
            i, j = base + half - 1, base + stride - 1
            if i < n_qubits and j < n_qubits:
                pairs.append((i, j))
        level += 1
    return pairs


class HybridQuantumNet(nn.Module):
    """
    Hybrid Classical-Quantum Neural Network encapsulating a PyTorch neural network
    with a PennyLane Variational Quantum Circuit (VQC).

    If `noise=True`, the circuit is executed on the `default.mixed` device with
    depolarizing channels modelling IBM Heron r2 calibration data.

    Repair log (checklist item 3.7):
      * multi-class head: `n_classes` maps the `n_readout` expectation values
        to any number of logits (needed for class-IL in E2; 2 by default).
      * general TTN: `ttn_pairs()` builds the merge schedule for any
        n_qubits >= 2 (the old hard-coded wiring crashed for n != 4).
      * functional freeze: `freeze_ansatz_layers()` freezes real parameters
        (the old `if '0' in name` check never matched `vqc.weights`,
        `fc.weight` or `fc.bias`, so nothing was ever frozen).
    """

    def __init__(self, ansatz='A', n_qubits=4, n_layers=2, noise=False,
                 noise_params=None, n_classes=2, n_readout=2,
                 ttn_readout='limited'):
        super(HybridQuantumNet, self).__init__()
        if n_qubits < 2:
            raise ValueError("n_qubits must be >= 2")
        if ttn_readout not in ('limited', 'root'):
            raise ValueError("ttn_readout must be 'limited' or 'root'")
        self.n_qubits = n_qubits
        self.ansatz = ansatz
        self.n_layers = n_layers
        self.noise = noise
        self.noise_params = noise_params if noise_params is not None else HERON_R2_NOISE
        self.n_classes = n_classes
        self.n_readout = n_readout
        self.ttn_readout = ttn_readout

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
            self.ttn_block_pairs = ttn_pairs(n_qubits)
            self.weight_shapes = {"weights": (n_layers, len(self.ttn_block_pairs), 2)}
        else:
            raise ValueError("Ansatz must be 'A', 'B', or 'C'.")

        self.vqc = qml.qnn.TorchLayer(self._qnode(), self.weight_shapes)
        self.fc = nn.Linear(self.n_readout, n_classes)

    # ------------------------------------------------------------------
    # Functional freeze (repair of audit A2: the old `if '0' in name` filter
    # never matched any real parameter name and silently froze nothing).
    # ------------------------------------------------------------------
    def freeze_ansatz_layers(self, layer_indices):
        """Freeze blocks of the ansatz (rows of `vqc.weights`) by real name.

        The VQC parameters live in the single tensor `vqc.weights` with shape
        (n_layers, ...), so "the lower block" is row 0 (or the rows given in
        `layer_indices`). Freezing is functional (a gradient mask) because the
        rows of one tensor cannot carry independent `requires_grad` flags.

        Returns the list of names actually matched, so callers can assert the
        freeze took effect (self-test in self_tests.py).
        """
        layer_indices = [int(i) for i in layer_indices]
        for i in layer_indices:
            if not (0 <= i < self.n_layers):
                raise ValueError(f"layer index {i} out of range [0, {self.n_layers})")
        mask = torch.ones_like(self.vqc.weights)
        for i in layer_indices:
            mask[i] = 0.0
        self.register_buffer("freeze_mask", mask, persistent=False)
        self.vqc.weights.register_hook(lambda g: g * self.freeze_mask)
        self._frozen_layers = tuple(layer_indices)
        return [f"vqc.weights[{i}]" for i in layer_indices]

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

    def _ttn_noisy_or_not(self, weights):
        """Generalized TTN blocks (RY + RZ + CNOT), any n_qubits >= 2.

        For n_qubits=4 this reproduces the historical circuit exactly:
        blocks (0,1), (2,3) then (1,3), RY on the control, RZ on the target.
        """
        n_layers = weights.shape[0]
        for l in range(n_layers):
            for block_idx, (i, j) in enumerate(self.ttn_block_pairs):
                qml.RY(weights[l, block_idx, 0], wires=i)
                self._noise_1q(i)
                qml.RZ(weights[l, block_idx, 1], wires=j)
                self._noise_1q(j)
                qml.CNOT(wires=[i, j])
                self._noise_2q([i, j])

    def _readout_wires(self):
        if self.ansatz == 'C' and self.ttn_readout == 'root':
            return list(self.ttn_block_pairs[-1])
        return [0, 1]

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
                self._ttn_noisy_or_not(weights)

            # 3. Readout error + measurement
            self._noise_readout()
            return [qml.expval(qml.PauliZ(i)) for i in self._readout_wires()[:self.n_readout]]

        return circuit

    def forward(self, x):
        out = self.vqc(x)
        out = self.fc(out)
        return out
