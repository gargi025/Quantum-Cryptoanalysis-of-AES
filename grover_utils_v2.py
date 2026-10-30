"""Utilities for toy Grover search and quantum-aware AES risk assessment.

Important: the Grover oracle in this project marks a known target state directly.
It is not a reversible AES oracle and must not be described as an AES key-recovery attack.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List

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
    if candidate < 0 or candidate > 255:
        raise ValueError("candidate must fit in one byte for this toy AES baseline")
    return bytes([candidate]).ljust(16, b"\x00")


def classical_bruteforce_aes(key_bits: int, target_key: int, plaintext: bytes) -> BruteForceResult:
    """Brute-force a tiny candidate space using AES-ECB as a toy baseline.

    The AES key itself remains 128 bits; candidates vary only the first byte while
    all remaining bytes are zero. This is not brute force over arbitrary AES keys.
    """
    if AES is None:
        raise ImportError("Install pycryptodome: pip install pycryptodome")
    if not (1 <= key_bits <= 8):
        raise ValueError("Toy AES baseline supports key_bits from 1 through 8")
    if not (0 <= target_key < 2**key_bits):
        raise ValueError("target_key must fit inside key_bits")
    if len(plaintext) != 16:
        raise ValueError("For ECB demo, plaintext must be exactly 16 bytes")

    ciphertext = AES.new(padded_aes_key(target_key), AES.MODE_ECB).encrypt(plaintext)
    start = time.perf_counter()
    attempts = 0
    found = -1
    for candidate in range(2**key_bits):
        attempts += 1
        trial = AES.new(padded_aes_key(candidate), AES.MODE_ECB)
        if trial.encrypt(plaintext) == ciphertext:
            found = candidate
            break
    return BruteForceResult(key_bits, target_key, found, attempts, time.perf_counter() - start)


def optimal_grover_iterations(n_qubits: int) -> int:
    """Rounded optimal Grover iterations for one marked item in N=2^n."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    return max(1, round((math.pi / 4) * math.sqrt(2**n_qubits)))


def _target_bits_little_endian(target: int, n_qubits: int) -> str:
    return format(target, f"0{n_qubits}b")[::-1]


def apply_phase_oracle(qc: QuantumCircuit, target: int, n_qubits: int) -> None:
    """Mark one known basis state; this is a toy oracle, not reversible AES."""
    bits_le = _target_bits_little_endian(target, n_qubits)
    for i, bit in enumerate(bits_le):
        if bit == "0":
            qc.x(i)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for i, bit in enumerate(bits_le):
        if bit == "0":
            qc.x(i)


def apply_diffuser(qc: QuantumCircuit, n_qubits: int) -> None:
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


def build_grover_circuit(
    n_qubits: int,
    target: int,
    iterations: int | None = None,
    measure: bool = True,
) -> QuantumCircuit:
    """Create a Grover circuit for a toy keyspace of 2^n_qubits states."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    if not (0 <= target < 2**n_qubits):
        raise ValueError("target must fit in n_qubits")
    if iterations is None:
        iterations = optimal_grover_iterations(n_qubits)
    if iterations < 0:
        raise ValueError("iterations must be non-negative")

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


def run_counts(
    qc: QuantumCircuit,
    shots: int = 1024,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
) -> Dict[str, int]:
    """Run a circuit on AerSimulator and return measurement counts."""
    if shots < 1:
        raise ValueError("shots must be positive")
    simulator = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    tqc = transpile(qc, simulator, seed_transpiler=seed)
    result = simulator.run(tqc, shots=shots).result()
    return result.get_counts()


def qiskit_count_key(target: int, n_qubits: int) -> str:
    return format(target, f"0{n_qubits}b")


def success_probability(counts: Dict[str, int], target: int, n_qubits: int) -> float:
    key = qiskit_count_key(target, n_qubits)
    return counts.get(key, 0) / max(1, sum(counts.values()))


@dataclass
class NoisySweepResult:
    mean: float
    std: float
    per_seed: List[float]


def run_noisy_multiseed(
    n_qubits: int,
    target: int,
    noise_model: NoiseModel | None,
    shots: int = 1024,
    n_seeds: int = 10,
    iterations: int | None = None,
) -> NoisySweepResult:
    """Estimate run-to-run variability using independent simulator seeds."""
    import numpy as np

    if n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    qc = build_grover_circuit(n_qubits, target, iterations=iterations, measure=True)
    probs = [
        success_probability(
            run_counts(qc, shots=shots, noise_model=noise_model, seed=seed),
            target,
            n_qubits,
        )
        for seed in range(n_seeds)
    ]
    arr = np.asarray(probs)
    return NoisySweepResult(mean=float(arr.mean()), std=float(arr.std()), per_seed=probs)


def make_noise_model(kind: str, p: float) -> NoiseModel:
    """Create depolarizing, bit-flip, phase-flip, or no-noise model."""
    kind = kind.lower().strip().replace("-", "_").replace(" ", "_")
    if not 0 <= p <= 1:
        raise ValueError("Noise probability p must be between 0 and 1")

    noise_model = NoiseModel()
    if kind in ["none", "ideal", "no_noise"]:
        return noise_model

    if kind == "depolarizing":
        err1 = depolarizing_error(p, 1)
        err2 = depolarizing_error(p, 2)
    elif kind in ["bit_flip", "bitflip"]:
        err1 = pauli_error([("X", p), ("I", 1 - p)])
        err2 = err1.tensor(err1)
    elif kind in ["phase_flip", "phaseflip"]:
        err1 = pauli_error([("Z", p), ("I", 1 - p)])
        err2 = err1.tensor(err1)
    else:
        raise ValueError("kind must be depolarizing, bit_flip, phase_flip, or none")

    noise_model.add_all_qubit_quantum_error(err1, ["h", "x", "z", "sx", "rz"])
    noise_model.add_all_qubit_quantum_error(err2, ["cx", "cz", "mcx"])
    return noise_model


def grover_query_estimate(key_bits: int) -> float:
    """Idealized Grover query estimate (one marked item); not wall-clock runtime."""
    if key_bits < 1:
        raise ValueError("key_bits must be positive")
    return (math.pi / 4.0) * (2 ** (key_bits / 2))


def synthetic_medical_records() -> list[dict]:
    """Return fictional records for a non-sensitive application demonstration."""
    return [
        {"record_id": "SYN-001", "age": 34, "visit_type": "routine_checkup", "diagnosis_code": "Z00.00"},
        {"record_id": "SYN-002", "age": 58, "visit_type": "cardiology_followup", "diagnosis_code": "I10"},
        {"record_id": "SYN-003", "age": 22, "visit_type": "orthopedic_consult", "diagnosis_code": "M25.50"},
        {"record_id": "SYN-004", "age": 71, "visit_type": "diabetes_followup", "diagnosis_code": "E11.9"},
        {"record_id": "SYN-005", "age": 46, "visit_type": "imaging_review", "diagnosis_code": "R51.9"},
    ]


def aes_gcm_round_trip(payload: bytes, key_bits: int) -> dict:
    """Classically encrypt/decrypt payload with AES-GCM and verify its tag."""
    if AES is None:
        raise ImportError("Install pycryptodome: pip install pycryptodome")
    if key_bits not in (128, 256):
        raise ValueError("key_bits must be 128 or 256")
    from Crypto.Random import get_random_bytes

    key = get_random_bytes(key_bits // 8)
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(payload)
    decryptor = AES.new(key, AES.MODE_GCM, nonce=cipher.nonce)
    recovered = decryptor.decrypt_and_verify(ciphertext, tag)
    return {
        "AES key size": f"AES-{key_bits}",
        "key bits": key_bits,
        "nonce bytes": len(cipher.nonce),
        "ciphertext bytes": len(ciphertext),
        "authentication tag bytes": len(tag),
        "round-trip verified": recovered == payload,
        "ciphertext differs from plaintext": ciphertext != payload,
    }
