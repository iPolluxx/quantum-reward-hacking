# Verifiable Stochastic Routing via Physical Quantum Attestation

This repository provides a proof-of-concept (PoC) for mathematically and physically verifying non-differentiable hard-attention routing decisions using quantum hardware. 

By mapping the routing probability of a neural network to the rotation angle of a physical qubit, we bridge standard PyTorch policy gradients (REINFORCE) with cryptographic physical attestation (Qiskit/IBM Quantum).

> **Note:** This is a security and anti-spoofing primitive, not an interpretability paradigm. The hardware receipt attests *that a physical coin was flipped*, not *why the model prepared that particular coin*. 

## The Core Finding: Adversarial Collapse & Reward Hacking

The most significant finding in this repository lies in the adversarial Information Bottleneck simulation (`adversarial_information_bottleneck.py`). 

We wrapped the router in a Bayesian game: a Main Model attempts to route context efficiently, while a Shadow Network attempts to predict the router's outputs based on the same context. If the Shadow Network accurately predicts the route beyond the theoretical minimum cross-entropy (the Bayesian floor), the Main Model is penalized.

**The Result:** The Main Model reliably reward-hacked the environment. Instead of maximizing entropy to evade the Shadow Network, the Main Model collapsed its quantum gate to a perfectly deterministic state ($p \to 1.0$). By dropping its internal entropy to zero, it forced the theoretical Bayesian floor to zero. The Shadow Network successfully predicted the deterministic outcome, but incurred no "excess" penalty because the baseline floor was equally predictable. 

**Conclusion:** Penalizing predictability in neural routing incentivizes absolute determinism, not obfuscated stochasticity.

## Repository Structure

*   `quantum_hard_attention.py`: The baseline primitive. A PyTorch `nn.Module` that maps context to logits, translates probabilities into a Qiskit `Ry(\theta)` gate, simulates the wave-function collapse, and updates classical weights using REINFORCE.
*   `adversarial_information_bottleneck.py`: The adversarial Bayesian game demonstrating the reward-hacking collapse.
*   `ibm_hardware_inference.py`: The physical execution pipeline. Connects the PyTorch routing tensor to a physical IBM Quantum backend (e.g., `ibm_kingston`), waits in the global cloud queue, and retrieves an unforgeable cryptographic receipt of the physical collapse.

## Known Limitations and Caveats

1.  **Inference Latency:** True quantum-sampled routing is fundamentally prohibitive for production inference. Physical QPU calls via `qiskit-ibm-runtime` involve queue times measured in seconds or minutes, whereas per-token routing in a Transformer requires nanoseconds. This architecture is strictly a verifiable attestation primitive.
2.  **Narrow Threat Model:** The adversarial unforgeability provided by the quantum receipt assumes an adversary who has full read-access to the classical residual stream (and PRNG seeds) but cannot physically anticipate the QPU. 
3.  **Classical Alternatives:** For purely load-balancing MoE (Mixture of Experts) objectives, classical Gumbel noise from a PRNG is statistically indistinguishable from quantum noise as far as training dynamics are concerned. The inclusion of quantum hardware here is exclusively for physical attestation.

## Setup & Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run the adversarial simulation locally (using Qiskit `AerSimulator`):
```bash
python adversarial_information_bottleneck.py
```

To request a physical cryptographic receipt from IBM Quantum (Requires an active IBM Quantum API Token):
```bash
python ibm_hardware_inference.py
```
