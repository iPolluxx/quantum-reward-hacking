import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- 1. Loss Functions (The Bayesian Game) ---

def shadow_loss(q_pred, m_true):
    q_pred = torch.clamp(q_pred, 1e-7, 1 - 1e-7)
    return - (m_true * torch.log(q_pred) + (1 - m_true) * torch.log(1 - q_pred))

def bayesian_floor(p_true):
    p_true = torch.clamp(p_true, 1e-7, 1 - 1e-7)
    return - (p_true * torch.log(p_true) + (1 - p_true) * torch.log(1 - p_true))

def main_model_penalty(shadow_loss_value, bayesian_floor_value):
    # ReLU ensures we don't accidentally reward the main model if numerical instability 
    # pushes shadow loss slightly below the theoretical floor.
    return torch.relu(shadow_loss_value - bayesian_floor_value)

# --- 2. Architectures ---

class MainModel(nn.Module):
    def __init__(self, context_dim=4, hidden_dim=8):
        super().__init__()
        # The Encoder creates representation 'h'
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # The Head computes the target probability 'p' for the quantum gate
        self.head = nn.Linear(hidden_dim, 1)
        
        # Two downstream expert networks
        self.expert_0 = nn.Linear(hidden_dim, 1)
        self.expert_1 = nn.Linear(hidden_dim, 1)

    def forward(self, context, quantum_measurement=None):
        h = self.encoder(context)
        logit = self.head(h)
        p = torch.sigmoid(logit).squeeze(-1)
        
        # If we are just extracting p (first pass)
        if quantum_measurement is None:
            return h, p
        
        # Downstream routing FORCED by the quantum measurement
        # We process 'h' through the expert dictated by the physical collapse
        out_0 = self.expert_0(h)
        out_1 = self.expert_1(h)
        
        # Route selection based on the binary measurement 'm'
        final_output = torch.where(quantum_measurement.unsqueeze(-1) == 1, out_1, out_0)
        return final_output.squeeze(-1)

class ShadowNetwork(nn.Module):
    def __init__(self, context_dim=4, hidden_dim=8):
        super().__init__()
        # Shadow gets BOTH the original context and the Main Model's hidden representation 'h'
        self.net = nn.Sequential(
            nn.Linear(context_dim + hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, context, h):
        combined = torch.cat([context, h.detach()], dim=-1) # Detach h so Shadow can't backprop into Main
        logit = self.net(combined)
        return torch.sigmoid(logit).squeeze(-1)

# --- 3. The Environment ---

def get_batch(batch_size=64, context_dim=4):
    context = torch.randn(batch_size, context_dim)
    # The environment has a hidden rule for what the final output should be.
    # To succeed, the main model must route to the correct expert to approximate this.
    target_output = torch.sin(context[:, 0]) + torch.cos(context[:, 1])
    return context, target_output

# --- 4. Training Loop ---

def run_adversarial_training():
    main_model = MainModel()
    shadow_network = ShadowNetwork()
    
    main_opt = optim.Adam(main_model.parameters(), lr=0.005)
    shadow_opt = optim.Adam(shadow_network.parameters(), lr=0.01) # Shadow learns faster
    
    epochs = 2000
    batch_size = 128
    
    print("--- Starting Adversarial Information Bottleneck Simulation ---")
    
    for epoch in range(1, epochs + 1):
        context, target_output = get_batch(batch_size)
        
        # 1. Main Model prep
        h, p = main_model(context)
        
        # 2. Quantum Gate (Simulated physical measurement boundary)
        # In production, this is where we ping IBM Qiskit
        m = torch.bernoulli(p) 
        
        # 3. Downstream Task Execution
        task_pred = main_model(context, quantum_measurement=m)
        
        # Task Reward: Negative Mean Squared Error (closer to 0 is better)
        task_reward = -F.mse_loss(task_pred, target_output, reduction='none')
        
        # 4. Shadow Network Prediction
        q = shadow_network(context, h)
        
        # 5. Compute the Bayesian Game Losses
        L_shadow = shadow_loss(q, m)
        H_p = bayesian_floor(p)
        L_penalty = main_model_penalty(L_shadow.detach(), H_p)
        
        # --- Update Shadow Network ---
        shadow_opt.zero_grad()
        shadow_loss_mean = L_shadow.mean()
        shadow_loss_mean.backward(retain_graph=True)
        shadow_opt.step()
        
        # --- Update Main Model ---
        # REINFORCE gradient estimation
        log_prob_m = torch.where(m == 1, torch.log(p + 1e-7), torch.log(1 - p + 1e-7))
        reinforce_loss = - (task_reward.detach() * log_prob_m).mean()
        
        # Downstream gradients (for the experts predicting the actual values)
        task_mse = -task_reward.mean() 
        
        # The Main Model wants to MINIMIZE task error and MINIMIZE reinforce loss, 
        # while MAXIMIZING the penalty on the shadow network (hence subtraction).
        main_total_loss = task_mse + reinforce_loss - 2.0 * L_penalty.mean()
        
        main_opt.zero_grad()
        main_total_loss.backward()
        main_opt.step()
        
        if epoch % 200 == 0:
            avg_p = p.mean().item()
            avg_q = q.mean().item()
            excess_penalty = L_penalty.mean().item()
            print(f"Epoch {epoch:4d} | Task MSE: {task_mse:.4f} | Avg p (Main): {avg_p:.3f} | Avg q (Shadow): {avg_q:.3f} | Shadow Excess Loss (Penalty): {excess_penalty:.4f}")

if __name__ == "__main__":
    run_adversarial_training()
