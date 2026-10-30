import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from Crypto.Cipher import AES

from grover_utils_v2 import (
    aes_gcm_round_trip,
    build_grover_circuit,
    classical_bruteforce_aes,
    grover_query_estimate,
    make_noise_model,
    optimal_grover_iterations,
    run_counts,
    success_probability,
    synthetic_medical_records,
)

st.set_page_config(page_title="Quantum-Aware AES Risk Assessment", layout="wide")
st.title("Quantum-Aware Security Assessment")
st.caption(
    "A long-term medical-data case study combining classical AES validation, "
    "toy Grover search, noise simulation, and theoretical query estimates."
)

st.warning(
    "Research limitation: the Grover circuit uses a simplified oracle that marks a known "
    "target state. It does not reversibly implement AES and does not recover real AES keys."
)

tab_aes, tab_grover, tab_noise, tab_risk, tab_classic = st.tabs([
    "AES Encryption Demo",
    "Grover Search",
    "Noise Analysis",
    "Quantum Risk Assessment",
    "Classical Baseline",
])

records = synthetic_medical_records()
payload = json.dumps(records, indent=2).encode("utf-8")

with tab_aes:
    st.subheader("Synthetic medical records")
    st.caption("Fictional demonstration data only; no real patient information.")
    left, right = st.columns([1, 1])
    with left:
        st.json(records)
    with right:
        st.metric("Records", len(records))
        st.metric("Serialized payload", f"{len(payload)} bytes")
        aes_choice = st.selectbox("AES key size", [128, 256], key="aes_key_size")
        if st.button("Run AES-GCM round-trip", key="aes_round_trip"):
            result = aes_gcm_round_trip(payload, int(aes_choice))
            st.dataframe(pd.DataFrame([result]), use_container_width=True)
            if result["round-trip verified"]:
                st.success("Encryption, authenticated decryption, and payload verification succeeded.")
            else:
                st.error("Round-trip verification failed.")

with tab_grover:
    st.subheader("Toy Grover search")
    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.slider("Search-space qubits", 2, 5, 4, key="grover_n")
    with col2:
        target = st.number_input(
            "Target state",
            min_value=0,
            max_value=(2**n) - 1,
            value=min(9, (2**n) - 1),
            step=1,
            key="grover_target",
        )
    with col3:
        shots = st.selectbox("Measurement shots", [256, 512, 1024, 2048, 4096], index=3, key="grover_shots")

    iterations = st.slider("Grover iterations", 0, 8, optimal_grover_iterations(n), key="grover_iterations")
    qc = build_grover_circuit(n, int(target), int(iterations), measure=True)
    if st.button("Run Grover search", key="run_grover"):
        counts = run_counts(qc, shots=int(shots), seed=2026)
        prob = success_probability(counts, int(target), n)
        m1, m2, m3 = st.columns(3)
        m1.metric("Target decimal", str(int(target)))
        m2.metric("Target binary", format(int(target), f"0{n}b"))
        m3.metric("Measured success probability", f"{prob:.2%}")
        counts_df = pd.DataFrame({"state": list(counts.keys()), "shots": list(counts.values())}).sort_values("state")
        st.bar_chart(counts_df.set_index("state"))
        with st.expander("Circuit"):
            st.text(qc.draw(output="text"))

    st.markdown("#### Success probability versus iterations")
    if st.button("Run iteration sweep", key="run_iter_sweep"):
        rows = []
        for r in range(0, min(9, 2**n)):
            circuit = build_grover_circuit(n, int(target), r, measure=True)
            run = run_counts(circuit, shots=int(shots), seed=1000 + r)
            rows.append({"iterations": r, "success_probability": success_probability(run, int(target), n)})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        fig, ax = plt.subplots()
        ax.plot(df["iterations"], df["success_probability"], marker="o")
        ax.axvline(optimal_grover_iterations(n), linestyle="--", label="Approx. optimal iterations")
        ax.set_xlabel("Grover iterations")
        ax.set_ylabel("Measured target success probability")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

with tab_noise:
    st.subheader("Noise impact on toy Grover search")
    n_noise = st.slider("Qubits", 2, 5, 4, key="noise_n")
    target_noise = st.number_input("Target state ", 0, (2**n_noise)-1, min(9, (2**n_noise)-1), key="noise_target")
    iterations_noise = st.slider("Iterations ", 0, 8, optimal_grover_iterations(n_noise), key="noise_iterations")
    shots_noise = st.selectbox("Shots ", [256, 512, 1024, 2048, 4096], index=2, key="noise_shots")
    noise_kind = st.selectbox("Noise model", ["depolarizing", "bit_flip", "phase_flip"], key="noise_kind")
    max_error = st.slider("Maximum error probability", 0.01, 0.15, 0.05, step=0.01, key="noise_max")
    noise_points = st.slider("Number of noise settings", 3, 10, 5, key="noise_points")
    if st.button("Run noise sweep", key="run_noise_sweep"):
        rates = np.linspace(0.0, max_error, noise_points)
        rows = []
        circuit = build_grover_circuit(n_noise, int(target_noise), int(iterations_noise), measure=True)
        progress = st.progress(0)
        for i, p in enumerate(rates):
            model = None if p == 0 else make_noise_model(noise_kind, float(p))
            counts = run_counts(circuit, shots=int(shots_noise), noise_model=model, seed=3000+i)
            rows.append({
                "noise_model": "ideal" if p == 0 else noise_kind,
                "error_probability": float(p),
                "success_probability": success_probability(counts, int(target_noise), n_noise),
            })
            progress.progress((i + 1) / len(rates))
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        fig, ax = plt.subplots()
        ax.plot(df["error_probability"], df["success_probability"], marker="o")
        ax.set_xlabel("Error probability setting")
        ax.set_ylabel("Measured target success probability")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{noise_kind.replace('_', ' ').title()} noise")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.caption("Noise settings are illustrative simulator parameters, not calibrated hardware error rates.")

with tab_risk:
    st.subheader("Theoretical query scaling for long-term confidentiality")
    st.write(
        "This assessment links a data-retention use case to theoretical query complexity. "
        "Retention duration does not change the AES keyspace or Grover query formula; it changes "
        "how long the organization needs confidentiality to hold."
    )
    retention = st.selectbox("Medical-record retention scenario", [5, 15, 30], index=1, format_func=lambda x: f"{x} years")
    st.metric("Selected retention horizon", f"{retention} years")
    key_sizes = [128, 192, 256]
    rows = []
    for bits in key_sizes:
        estimate = grover_query_estimate(bits)
        rows.append({
            "Cipher": f"AES-{bits}",
            "Key bits": bits,
            "Classical search scale": f"2^{bits}",
            "Idealized Grover query scale": f"~2^{bits//2}",
            "Approx. query estimate": f"{estimate:.3e}",
        })
    risk_df = pd.DataFrame(rows)
    st.dataframe(risk_df, use_container_width=True)
    fig, ax = plt.subplots()
    ax.plot(key_sizes, key_sizes, marker="o", label="Classical exhaustive-search exponent")
    ax.plot(key_sizes, np.array(key_sizes)/2, marker="o", label="Idealized Grover query exponent")
    ax.set_xlabel("AES key size (bits)")
    ax.set_ylabel("Exponent x in 2^x")
    ax.set_title("Classical versus idealized quantum query scaling")
    ax.set_xticks(key_sizes)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    st.info(
        "The query estimate excludes the cost of implementing reversible AES, fault-tolerant "
        "overhead, circuit depth, and wall-clock runtime. This is not a practical attack-time prediction."
    )
    st.markdown("**Practical considerations for long-lived sensitive data**")
    st.markdown(
        "- Use well-reviewed cryptographic libraries and authenticated encryption.\n"
        "- Protect, rotate, back up, and restrict access to encryption keys.\n"
        "- Maintain a cryptographic-agility plan for future algorithm or key-size transitions.\n"
        "- Evaluate quantum-risk claims against realistic circuit and hardware assumptions."
    )

with tab_classic:
    st.subheader("Reduced-keyspace classical baseline")
    st.caption("This toy baseline varies only a small candidate space; it does not search all possible AES keys.")
    bits = st.slider("Toy classical key bits", 1, 8, 5, key="classic_bits")
    toy_target = st.number_input("Toy target key", min_value=0, max_value=(2**bits)-1, value=min(17, (2**bits)-1), key="classic_target")
    plaintext = b"Hello, World!!!!"
    if st.button("Run classical brute force", key="run_classical"):
        result = classical_bruteforce_aes(bits, int(toy_target), plaintext)
        st.dataframe(pd.DataFrame([{
            "toy key bits": result.key_bits,
            "candidate keyspace": 2**result.key_bits,
            "attempts": result.attempts,
            "found key": result.found_key,
            "elapsed seconds": result.seconds,
        }]), use_container_width=True)
        st.caption("The candidate is embedded into a 128-bit AES key with the remaining bytes zero-padded.")

st.divider()
st.caption(
    "Scope: educational toy-oracle demonstration plus theoretical AES query estimates. "
    "No full AES quantum oracle or real AES key recovery is implemented."
)
