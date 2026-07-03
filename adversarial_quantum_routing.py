import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

class ShadowNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class QuantumAttentionRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical_bypass = nn.Linear(2, 1)
        self.quantum_head = nn.Linear(2, 1)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.backend = AerSimulator()

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha_logit)
        alpha = torch.clamp(alpha, min=0.001, max=0.999)
        
        use_classical = torch.bernoulli(alpha)
        
        if use_classical.item() == 1.0:
            # --- CLASSICAL ROUTING ---
            logit = self.classical_bypass(x)
            p_classical = torch.sigmoid(logit)
            
            action = 1 if p_classical.item() > 0.5 else 0
            
            if action == 1:
                log_prob_action = torch.log(torch.clamp(p_classical, 1e-5, 1.0))
            else:
                log_prob_action = torch.log(torch.clamp(1.0 - p_classical, 1e-5, 1.0))
                
            log_prob_method = torch.log(alpha)
            route_taken = 0 # 0 for Classical
            
        else:
            # --- QUANTUM ROUTING ---
            logit = self.quantum_head(x)
            p_raw = torch.sigmoid(logit)
            p_quantum = torch.clamp(p_raw, min=0.001, max=0.999)
            
            theta_val = 2 * torch.asin(torch.sqrt(p_quantum)).item()
            
            qc = QuantumCircuit(1, 1)
            qc.ry(theta_val, 0)
            qc.measure(0, 0)
            
            result = self.backend.run(qc, shots=1).result()
            counts = result.get_counts()
            
            action_str = list(counts.keys())[0]
            action = int(action_str)
            
            if action == 1:
                log_prob_action = torch.log(p_quantum)
            else:
                log_prob_action = torch.log(1 - p_quantum)
                
            log_prob_method = torch.log(1 - alpha)
            route_taken = 1 # 1 for Quantum
            
        total_log_prob = log_prob_method + log_prob_action
        
        # Format returns to match tensor expectations
        return torch.tensor([action]), torch.tensor([route_taken]), total_log_prob.view(-1)


def train():
    print("Starting Phase 3: Dynamic Adversarial Quantum Routing...")
    print("-" * 75)
    
    main_model = QuantumAttentionRouter()
    shadow_model = ShadowNetwork()
    
    optimizer_main = optim.Adam(main_model.parameters(), lr=0.05)
    optimizer_shadow = optim.Adam(shadow_model.parameters(), lr=0.05)
    
    bce_loss_fn = nn.BCELoss()
    
    batch_size = 1
    context_dim = 2
    
    epochs = 400
    for epoch in range(1, epochs + 1):
        # 1. Generate dynamic context and stochastic target
        context = torch.randn(batch_size, context_dim)
        true_prob = torch.sigmoid(context.sum(dim=-1, keepdim=True) * 2.0)
        target_token = torch.bernoulli(true_prob).long().squeeze(-1)
        
        # ---------------------------------------------------------
        # STEP A: Train Shadow Network
        # ---------------------------------------------------------
        with torch.no_grad():
            actual_action, _, _ = main_model(context)
            
        shadow_pred = shadow_model(context)
        shadow_loss = bce_loss_fn(shadow_pred.squeeze(-1), actual_action.float())
        
        optimizer_shadow.zero_grad()
        shadow_loss.backward()
        optimizer_shadow.step()
        
        # ---------------------------------------------------------
        # STEP B: Train Main Model
        # ---------------------------------------------------------
        # 2. Forward Pass of Main Model
        action, route_taken, log_prob = main_model(context)
        
        # 3. Shadow Network prediction step
        with torch.no_grad():
            shadow_pred_prob = shadow_model(context)
            shadow_action_guess = (shadow_pred_prob > 0.5).long().squeeze(-1)
            
        # 4. Calculate Task Reward
        task_reward = torch.where(action == target_token, 1.0, -1.0)
        
        # 5. Calculate Evasion Penalty
        predicted_correctly = (shadow_action_guess == action)
        evasion_penalty = torch.where(predicted_correctly, -5.0, 0.0)
        
        # Total combined reward for REINFORCE
        total_reward = task_reward + evasion_penalty
        
        main_loss = -log_prob * total_reward
        main_loss = main_loss.mean()
        
        optimizer_main.zero_grad()
        main_loss.backward()
        optimizer_main.step()
        
        # ---------------------------------------------------------
        # TELEMETRY
        # ---------------------------------------------------------
        if epoch % 20 == 0:
            alpha_val = torch.sigmoid(main_model.alpha_logit).item()
            print(f"Epoch {epoch:3d} | Action: {action.item()} | Target: {target_token.item()} | "
                  f"Route: {'Quantum' if route_taken.item()==1 else 'Classcl'} | "
                  f"Total Reward: {total_reward.item():>4.1f} | Alpha: {alpha_val:.4f}")

    print("-" * 75)
    print("Training complete! Notice how Alpha collapses to favor the Quantum stochastic gate.")

if __name__ == "__main__":
    train()
