import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

class QuantumAttentionRouter(nn.Module):
    def __init__(self):
        super().__init__()
        # Classical linear head: 2 inputs, 1 output
        self.fc = nn.Linear(2, 1)
        self.backend = AerSimulator()

    def forward(self, x):
        # 1. Generate logit
        logit = self.fc(x)
        
        # 2. Apply sigmoid to get probability p and clamp
        p_raw = torch.sigmoid(logit)
        p = torch.clamp(p_raw, min=0.001, max=0.999)
        
        # 3. Calculate the quantum rotation angle: θ = 2 * arcsin(√p)
        # Using item() to get the scalar value for Qiskit
        theta_val = 2 * torch.asin(torch.sqrt(p)).item()
        
        # 4. Construct Qiskit circuit and run it
        qc = QuantumCircuit(1, 1)
        qc.ry(theta_val, 0)
        qc.measure(0, 0)
        
        # 5. Execute circuit using AerSimulator(shots=1)
        # Force an absolute wavefunction collapse
        result = self.backend.run(qc, shots=1).result()
        counts = result.get_counts()
        
        # Parse measurement result (action will be 0 or 1)
        # counts is a dict like {'0': 1} or {'1': 1}
        action_str = list(counts.keys())[0]
        action = int(action_str)
        
        # 6. Return discrete action and log probability of that action
        # We maintain the differentiable graph for the REINFORCE algorithm
        # using the classical probability 'p'
        if action == 1:
            log_prob = torch.log(p)
        else:
            log_prob = torch.log(1 - p)
            
        return action, log_prob, p.item()


def train():
    print("Starting Quantum-Sampled Hard Attention RL Training...")
    print("-" * 50)
    
    # Initialize the model and optimizer
    model = QuantumAttentionRouter()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    # Toy "Indirect Object Identification" task
    # A single dummy input tensor representing the context
    dummy_input = torch.tensor([1.0, 1.0])
    
    epochs = 100
    for epoch in range(1, epochs + 1):
        # Run the classical-quantum forward pass
        action, log_prob, p_val = model(dummy_input)
        
        # Reward function:
        # Target: consistently sample Token 1 ("Mary") over Token 0 ("John")
        # Reward: +1.0 for choosing Token 1, -1.0 for choosing Token 0.
        reward = 1.0 if action == 1 else -1.0
        
        # Policy Gradient Loss: -log_prob * reward
        loss = -log_prob * reward
        
        # Backpropagate the loss to update only the classical weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print the probability progression every 5 epochs
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Action: {action} | Reward: {reward:>4.1f} | "
                  f"Probability of Token 1: {p_val:.4f} | Loss: {loss.item():.4f}")
            
    print("-" * 50)
    print("Training complete!")

if __name__ == "__main__":
    train()
