import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from grover_utils import (
    build_grover_circuit,
    classical_bruteforce_aes,
    make_noise_model,
    optimal_grover_iterations,
    run_counts,
    success_probability,
)

st.set_page_config(page_title="Quantum Threat to AES: Grover Demo", layout="wide")
st.title("Quantum Key Search Demo: Grover's Algorithm vs Classical AES Brute Force")
st.caption("Toy 2–5 bit keyspace demo with extrapolation to AES-128 and AES-256")

with st.sidebar:
    st.header("Demo controls")
    n = st.slider("Toy key size (qubits)", 2, 5, 3)
    target = st.number_input("Correct toy key", min_value=0, max_value=2**n - 1, value=min(3, 2**n - 1))
    iterations = st.slider("Grover iterations", 0, max(1, 6), optimal_grover_iterations(n))
    shots = st.selectbox("Shots", [256, 512, 1024, 2048], index=2)
    noise_type = st.selectbox(
        "Noise model",
        ["None", "Depolarizing", "Bit-flip", "Phase-flip"]
    )

    error_rate = st.slider(
        "Gate error rate",
        min_value=0.0,
        max_value=0.15,
        value=0.0,
        step=0.01
    )
plaintext = b"Hello, World!!!!"  # exactly 16 bytes

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Classical brute force baseline")
    if st.button("Run classical brute force"):
        rows = []
        for bits in range(1, min(9, n + 5)):
            tgt = min(target, 2**bits - 1)
            res = classical_bruteforce_aes(bits, tgt, plaintext)
            rows.append({"key_bits": bits, "keyspace": 2**bits, "attempts": res.attempts, "time_sec": res.seconds})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        fig, ax = plt.subplots()
        ax.plot(df["key_bits"], df["time_sec"], marker="o")
        ax.set_xlabel("Toy key size (bits)")
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Classical brute-force time grows linearly with keyspace")
        ax.grid(True)
        st.pyplot(fig)

with col2:
    st.subheader("2. Grover quantum search")

    qc = build_grover_circuit(n, int(target), int(iterations))

    noise_map = {
        "None": None,
        "Depolarizing": "depolarizing",
        "Bit-flip": "bit_flip",
        "Phase-flip": "phase_flip",
    }

    selected_noise = noise_map[noise_type]

    if selected_noise is None or error_rate == 0.0:
        noise_model = None
        st.info("Running ideal/noiseless Grover simulation.")
    else:
        noise_model = make_noise_model(selected_noise, float(error_rate))
        st.warning(f"Running noisy simulation: {noise_type}, error rate = {error_rate:.2f}")

    counts = run_counts(qc, shots=shots, noise_model=noise_model)
    prob = success_probability(counts, int(target), n)

    target_binary = format(int(target), f"0{n}b")

    st.metric("Target key", f"{int(target)} = {target_binary}")
    st.metric("Success probability", f"{prob:.3f}")

    counts_df = pd.DataFrame.from_dict(counts, orient="index", columns=["shots"])
    counts_df = counts_df.sort_index()
    st.bar_chart(counts_df)

    with st.expander("Show circuit text"):
        st.text(qc.draw(output="text"))

st.subheader("3. Success probability vs Grover iterations")
rows = []
noise_map = {
    "None": None,
    "Depolarizing": "depolarizing",
    "Bit-flip": "bit_flip",
    "Phase-flip": "phase_flip",
}

selected_noise = noise_map[noise_type]

if selected_noise is None or error_rate == 0.0:
    selected_noise_model = None
else:
    selected_noise_model = make_noise_model(selected_noise, float(error_rate))

for r in range(0, 7):
    qc_r = build_grover_circuit(n, int(target), r)
    counts_r = run_counts(qc_r, shots=shots, noise_model=selected_noise_model)
    rows.append({
        "iterations": r,
        "success_probability": success_probability(counts_r, int(target), n)
    })
df_iter = pd.DataFrame(rows)
fig, ax = plt.subplots()
ax.plot(df_iter["iterations"], df_iter["success_probability"], marker="o")
ax.axvline((math.pi / 4) * math.sqrt(2**n), linestyle="--", label="π/4 √N")
ax.set_xlabel("Grover iterations")
ax.set_ylabel("Success probability")
ax.set_ylim(0, 1.05)
ax.set_title("Grover has an optimal stopping point")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("4. Classical vs Grover complexity extrapolation")
key_sizes = np.arange(1, 257)
classical = 2.0 ** key_sizes
grover = 2.0 ** (key_sizes / 2)
fig, ax = plt.subplots()
ax.plot(key_sizes, np.log2(classical), label="Classical brute force: log2(2^n) = n")
ax.plot(key_sizes, np.log2(grover), label="Grover search: log2(2^(n/2)) = n/2")
ax.axvline(128, linestyle="--", label="AES-128")
ax.axvline(256, linestyle="--", label="AES-256")
ax.set_xlabel("AES key size (bits)")
ax.set_ylabel("log2 operations")
ax.set_title("Grover quadratically reduces exhaustive key search")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("5. Noise analysis")
noise_kinds = ["depolarizing", "bit_flip", "phase_flip"]
error_rates = np.linspace(0, 0.15, 10)
noise_rows = []
for kind in noise_kinds:
    for p in error_rates:
        nm = make_noise_model(kind, float(p))
        qc_noise = build_grover_circuit(n, int(target), int(iterations))
        counts_noise = run_counts(qc_noise, shots=shots, noise_model=nm)
        noise_rows.append({"noise_type": kind, "error_rate": p, "success_probability": success_probability(counts_noise, int(target), n)})
df_noise = pd.DataFrame(noise_rows)
fig, ax = plt.subplots()
for kind in noise_kinds:
    part = df_noise[df_noise["noise_type"] == kind]
    ax.plot(part["error_rate"], part["success_probability"], marker="o", label=kind)
ax.set_xlabel("Error rate per gate")
ax.set_ylabel("Success probability")
ax.set_ylim(0, 1.05)
ax.set_title("Grover performance under quantum noise")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("Conclusion")
st.write(
    "This demo uses a simplified oracle for a reduced keyspace. A real AES Grover oracle must implement AES reversibly, "
    "which is too large for a laptop simulation. The takeaway is theoretical: Grover changes exhaustive key search from "
    "O(N) to O(√N), so an n-bit symmetric key gives roughly n/2 bits of quantum search security."
)
