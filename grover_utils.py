"""Utility functions for a toy AES key-search demonstration using Grover's algorithm.

This project intentionally uses a simplified oracle: instead of implementing the full
AES encryption function as a reversible quantum circuit, the oracle marks the known toy
key. This keeps the circuit small enough to run on a laptop while still demonstrating
Grover amplitude amplification.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

try:
    from Crypto.Cipher import AES
except Exception:  # pragma: no cover
    AES = None

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error


@dataclass
class BruteForceResult:
    key_bits: int
    target_key: int
    found_key: int
    attempts: int
    seconds: float


def padded_aes_key(candidate: int) -> bytes:
    """Represent a toy integer key as a valid 16-byte AES key."""
    return bytes([candidate]).ljust(16, b"\x00")


def classical_bruteforce_aes(key_bits: int, target_key: int, plaintext: bytes) -> BruteForceResult:
    """Brute-force a tiny AES keyspace: keys 0..2^key_bits-1.

    The real AES key is still 128 bits long, but only the first byte is varied and the
    remaining bytes are zero-padded. This is a toy model for demonstration.
    """
    if AES is None:
        raise ImportError("Install pycryptodome: pip install pycryptodome")
    if not (0 <= target_key < 2**key_bits):
        raise ValueError("target_key must fit inside key_bits")
    if len(plaintext) != 16:
        raise ValueError("For ECB demo, plaintext must be exactly 16 bytes")

    cipher = AES.new(padded_aes_key(target_key), AES.MODE_ECB)
    ciphertext = cipher.encrypt(plaintext)

    start = time.perf_counter()
    attempts = 0
    found = -1
    for candidate in range(2**key_bits):
        attempts += 1
        trial = AES.new(padded_aes_key(candidate), AES.MODE_ECB)
        if trial.encrypt(plaintext) == ciphertext:
            found = candidate
            break
    seconds = time.perf_counter() - start
    return BruteForceResult(key_bits, target_key, found, attempts, seconds)


def optimal_grover_iterations(n_qubits: int) -> int:
    """Rounded optimal Grover iterations for one marked item in N=2^n."""
    return max(1, round((math.pi / 4) * math.sqrt(2**n_qubits)))


def _target_bits_little_endian(target: int, n_qubits: int) -> str:
    """Qiskit qubit order helper: qubit 0 is the least significant bit."""
    return format(target, f"0{n_qubits}b")[::-1]


def apply_phase_oracle(qc: QuantumCircuit, target: int, n_qubits: int) -> None:
    """Flip the phase of one computational basis state |target>.

    This implements a toy key-check oracle. For a real AES Grover attack, this block
    would need to reversibly compute AES_k(plaintext), compare it with ciphertext, and
    uncompute all temporary registers.
    """
    bits_le = _target_bits_little_endian(target, n_qubits)

    # Convert the target state to |11...1>.
    for i, bit in enumerate(bits_le):
        if bit == "0":
            qc.x(i)

    # Multi-controlled Z using H on the last qubit and MCX.
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    # Restore original basis.
    for i, bit in enumerate(bits_le):
        if bit == "0":
            qc.x(i)


def apply_diffuser(qc: QuantumCircuit, n_qubits: int) -> None:
    """Grover diffusion operator: inversion about the mean."""
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


def build_grover_circuit(n_qubits: int, target: int, iterations: int | None = None, measure: bool = True) -> QuantumCircuit:
    """Create a Grover circuit for a toy keyspace of 2^n_qubits keys."""
    if not (0 <= target < 2**n_qubits):
        raise ValueError("target must fit in n_qubits")
    if iterations is None:
        iterations = optimal_grover_iterations(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits if measure else 0)
    qc.h(range(n_qubits))
    qc.barrier(label="superposition")

    for _ in range(iterations):
        apply_phase_oracle(qc, target, n_qubits)
        qc.barrier(label="oracle")
        apply_diffuser(qc, n_qubits)
        qc.barrier(label="diffusion")

    if measure:
        qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run_counts(qc: QuantumCircuit, shots: int = 1024, noise_model: NoiseModel | None = None) -> Dict[str, int]:
    """Run a circuit on AerSimulator and return measurement counts."""
    simulator = AerSimulator(noise_model=noise_model)
    tqc = transpile(qc, simulator)
    result = simulator.run(tqc, shots=shots).result()
    return result.get_counts()


def qiskit_count_key(target: int, n_qubits: int) -> str:
    """Return the measurement key string used by Qiskit counts for an integer target."""
    # Qiskit prints classical bitstrings with highest classical bit on the left.
    return format(target, f"0{n_qubits}b")


def success_probability(counts: Dict[str, int], target: int, n_qubits: int) -> float:
    key = qiskit_count_key(target, n_qubits)
    return counts.get(key, 0) / max(1, sum(counts.values()))


def make_noise_model(kind: str, p: float) -> NoiseModel:
    """Create depolarizing, bit-flip, or phase-flip noise model."""
    noise_model = NoiseModel()
    if kind == "depolarizing":
        err1 = depolarizing_error(p, 1)
        err2 = depolarizing_error(p, 2)
    elif kind == "bit_flip":
        err1 = pauli_error([("X", p), ("I", 1 - p)])
        err2 = err1.tensor(err1)
    elif kind == "phase_flip":
        err1 = pauli_error([("Z", p), ("I", 1 - p)])
        err2 = err1.tensor(err1)
    else:
        raise ValueError("kind must be depolarizing, bit_flip, or phase_flip")

    # These are the gates used after transpilation for small Grover circuits.
    noise_model.add_all_qubit_quantum_error(err1, ["h", "x", "z", "sx", "rz"])
    noise_model.add_all_qubit_quantum_error(err2, ["cx", "cz"])
    return noise_model
