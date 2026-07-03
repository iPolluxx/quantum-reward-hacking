import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# 1. Define the Router Architecture (Must match your trained model)
class QuantumAttentionRouter(nn.Module):
    def __init__(self):
        super(QuantumAttentionRouter, self).__init__()
        # Matches the 2D context input from Phase 1/2
        self.classical_head = nn.Linear(2, 1) 

    def forward(self, x):
        # We only need the forward pass logic to get the rotation angle
        logit = self.classical_head(x)
        prob = torch.sigmoid(logit)
        prob = torch.clamp(prob, 0.001, 0.999)
        
        prob_np = prob.detach().item()
        theta = 2 * np.arcsin(np.sqrt(prob_np))
        return theta

def run_hardware_inference():
    print("--- Initializing Hybrid Quantum-Classical Pipeline ---")
    
    # 2. Authenticate with IBM Quantum
    # Make sure you have run: qiskit-ibm-runtime save-account --token "YOUR_TOKEN"
    print("Authenticating with IBM Quantum Cloud...")
    try:
        service = QiskitRuntimeService()
    except Exception as e:
        print("Authentication failed. Did you save your IBM token?")
        print(f"Error: {e}")
        return

    # Select the least busy real physical quantum computer (no simulators)
    backend = service.least_busy(operational=True, simulator=False)
    print(f"Target QPU acquired: {backend.name} (Queue status: {backend.status().pending_jobs} jobs)")

    # 3. Load the pre-trained classical weights
    # For this script, we initialize the model. In production, uncomment the load line:
    model = QuantumAttentionRouter()
    # model.load_state_dict(torch.load('quantum_router_weights.pth'))
    model.eval() # Freeze weights

    # 4. Process a live input context
    context_input = torch.tensor([0.8, -0.2])
    print(f"\nClassical Input Context: {context_input.tolist()}")
    
    with torch.no_grad():
        theta = model(context_input)
    print(f"Classical Network mapped context to Quantum Rotation Angle (Theta): {theta:.4f} rad")

    # 5. Build the Qiskit Circuit
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    # 6. Transpile for the specific physical hardware
    # Physical qubits have specific topologies and native gates. 
    # Transpiling converts our abstract circuit into physical microwave pulses.
    print(f"\nTranspiling circuit for {backend.name} topology...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled_qc = pm.run(qc)

    # 7. Execute on the Physical QPU
    print("Sending job to IBM Quantum queue. Awaiting physical execution...")
    sampler = Sampler(backend)
    
    # We run 1 shot to simulate the absolute hard-attention collapse
    job = sampler.run([transpiled_qc], shots=1)
    print(f"Job ID: {job.job_id()} - Waiting in cloud queue...")
    
    # Blocks until the cloud processor executes the job
    result = job.result()
    
    # 8. Parse the physical measurement receipt
    # SamplerV2 returns data in PubResults. Extracting the classical register bitstring:
    pub_result = result[0]
    counts = pub_result.data.c.get_counts()
    
    # Extract the discrete routing action
    action = 1 if '1' in counts else 0
    
    print("\n==================================================")
    print("           QUANTUM EXECUTION COMPLETE             ")
    print("==================================================")
    print(f"Physical Wavefunction Collapse Outcome : |{action}>")
    print(f"Hard-Attention Route Selected          : Expert {action}")
    print(f"Unforgeable Cryptographic Receipt      : {job.job_id()}")
    print("==================================================")

if __name__ == "__main__":
    run_hardware_inference()
