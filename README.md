# Quantum Computing Mini Project

## Title
**Quantum Cryptanalysis of AES using Grover’s Algorithm: Simulation, Noise Study, and Security Extrapolation**

## What this project shows
This project demonstrates how Grover's algorithm gives a quadratic speedup for brute-force key search. Since a full AES-128 Grover oracle needs thousands of logical/reversible operations and many qubits, the implementation uses a toy 2–5 bit keyspace and a simplified oracle, then extrapolates the result to AES-128 and AES-256.

## Demo options

### Option A: Streamlit demo
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Option B: Use utility functions in Python
```python
from grover_utils import build_grover_circuit, run_counts, success_probability
qc = build_grover_circuit(n_qubits=3, target=5, iterations=2)
counts = run_counts(qc, shots=1024)
print(counts)
print(success_probability(counts, target=5, n_qubits=3))
```

## Demonstration flow
1. Show classical AES brute force on tiny keyspaces.
2. Show Grover circuit for 2-bit or 3-bit toy key search.
3. Show histogram where the correct key has highest probability.
4. Show success probability vs number of Grover iterations.
5. Show complexity comparison: classical `2^n` vs Grover `2^(n/2)`.
6. Show noisy simulation with depolarizing, bit-flip, and phase-flip errors.
7. Explain limitation: the oracle is simplified; real AES oracle requires reversible AES.

## Conclusion
Grover's algorithm does not instantly break AES. It gives a quadratic speedup. So AES-128's exhaustive-search complexity changes from about 2^128 to about 2^64 quantum oracle calls in the ideal model, while AES-256 changes from 2^256 to about 2^128. This is why larger symmetric keys are preferred for long-term high-security contexts, although practical quantum attacks on AES are far beyond current hardware.
