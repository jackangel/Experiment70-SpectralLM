import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import glob
import random
from pathlib import Path
from typing import List, Tuple, Optional
from transformers import GPT2TokenizerFast
import pyarrow.parquet as pq

# ==========================================
# 1. THE WATCHER (HOMEOSTATIC REGULATOR)
# ==========================================

class HomeostaticRegulator:
    """
    PID Controller for Regularization.
    Maintains the 'Gap' between Train and Val loss near a target.
    """
    def __init__(self, target_gap=0.10, alpha=0.2):
        self.target_gap = target_gap
        self.alpha = alpha  # Smoothing factor
        
        self.train_ema = None
        self.val_ema = None
        
        # Current State
        self.current_dropout = 0.1
        self.current_wd = 0.01
        
        # Bounds
        self.min_drop, self.max_drop = 0.05, 0.6
        self.min_wd, self.max_wd = 1e-5, 0.3

    def update(self, train_loss, val_loss):
        # 1. Smooth the signals (Low-Pass Filter)
        if self.train_ema is None:
            self.train_ema = train_loss
            self.val_ema = val_loss
        else:
            self.train_ema = (1 - self.alpha) * self.train_ema + self.alpha * train_loss
            self.val_ema = (1 - self.alpha) * self.val_ema + self.alpha * val_loss

        # 2. Calculate the Overfitting Signal
        gap = self.val_ema - self.train_ema
        error = gap - self.target_gap
        
        # 3. PID Control (Proportional term mainly)
        k_dropout = 0.1  # Gain for dropout
        k_wd = 0.05      # Gain for weight decay

        self.current_dropout += (error * k_dropout)
        self.current_wd += (error * k_wd)

        # Clamp values
        self.current_dropout = max(self.min_drop, min(self.max_drop, self.current_dropout))
        self.current_wd = max(self.min_wd, min(self.max_wd, self.current_wd))

        # Determine Status Label
        if error > 0.05: status = "OVERFITTING"
        elif error < -0.05: status = "UNDERFITTING"
        else: status = "STABLE"

        return {
            "gap": gap,
            "dropout": self.current_dropout,
            "weight_decay": self.current_wd,
            "status": status
        }

# ==========================================
# 2. DATA LOADER & UTILS
# ==========================================

def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y

@torch.jit.script
def complex_mul(r1, i1, r2, i2):
    """Helper for complex multiplication: (r1+i1*j) * (r2+i2*j)"""
    real = r1 * r2 - i1 * i2
    imag = r1 * i2 + i1 * r2
    return real, imag

def parallel_gated_recurrence(u_real: torch.Tensor, u_imag: torch.Tensor, 
                              decay: torch.Tensor, 
                              rot_real: torch.Tensor, rot_imag: torch.Tensor,
                              state_real: torch.Tensor, state_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Parallel Scan implementation of the gated recurrence.
    Complexity: O(log L) steps instead of O(L).
    """
    B, L, D = u_real.shape
    
    # 1. Prepare Coefficients (The 'A' in h_t = A*h_{t-1} + u)
    # A = decay * e^(i*theta)
    # We combine decay and rotation into a single complex coefficient
    a_real = decay * rot_real
    a_imag = decay * rot_imag
    
    # 2. Prepare Inputs (The 'u' in h_t = A*h_{t-1} + u)
    # Initially, the accumulated input is just the input itself
    acc_u_real = u_real.clone()
    acc_u_imag = u_imag.clone()
    
    # 3. Prepare Accumulator for Coefficients
    acc_a_real = a_real.clone()
    acc_a_imag = a_imag.clone()
    
    # 4. Hillis-Steele Parallel Scan (Logarithmic Time)
    # We iterate 1, 2, 4, 8... until we cover the sequence length
    n_steps = int(math.ceil(math.log2(L)))
    
    for i in range(n_steps):
        stride = 1 << i
        
        # We need to shift tensors to align t with t-stride
        # Slicing creates the shift effectively
        
        # Get values from (t - stride)
        # We pad with zeros at the beginning because t < stride has no predecessor
        prev_a_r = torch.zeros_like(acc_a_real)
        prev_a_i = torch.zeros_like(acc_a_imag)
        prev_u_r = torch.zeros_like(acc_u_real)
        prev_u_i = torch.zeros_like(acc_u_imag)
        
        # Shifted assignment
        prev_a_r[:, stride:, :] = acc_a_real[:, :-stride, :]
        prev_a_i[:, stride:, :] = acc_a_imag[:, :-stride, :]
        prev_u_r[:, stride:, :] = acc_u_real[:, :-stride, :]
        prev_u_i[:, stride:, :] = acc_u_imag[:, :-stride, :]
        
        # --- The Associative Operator ---
        # (A_new, u_new) = (A_curr * A_prev,  A_curr * u_prev + u_curr)
        
        # 1. Update Coefficients: A_new = A_curr * A_prev
        new_a_r, new_a_i = complex_mul(acc_a_real, acc_a_imag, prev_a_r, prev_a_i)
        
        # 2. Update Inputs: u_new = A_curr * u_prev + u_curr
        # First calculate A_curr * u_prev
        prod_r, prod_i = complex_mul(acc_a_real, acc_a_imag, prev_u_r, prev_u_i)
        # Then add u_curr
        new_u_r = prod_r + acc_u_real
        new_u_i = prod_i + acc_u_imag
        
        # Update accumulators for next iteration
        acc_a_real = new_a_r
        acc_a_imag = new_a_i
        acc_u_real = new_u_r
        acc_u_imag = new_u_i

    # 5. Apply Initial State
    # The scan computed h_t assuming h_{-1} = 0. 
    # Now we add the contribution of the actual initial state h_{-1}.
    # Effect of initial state at time t is: (Prod_{0 to t} A) * state
    
    # Calculate A_cum * state
    # Note: acc_a_real/imag now holds the cumulative product of A for the whole window 0..t
    init_contribution_r, init_contribution_i = complex_mul(acc_a_real, acc_a_imag, 
                                                         state_real.unsqueeze(1), 
                                                         state_imag.unsqueeze(1))
    
    final_real = acc_u_real + init_contribution_r
    final_imag = acc_u_imag + init_contribution_i
    
    # 6. Extract final states for the next batch
    last_state_real = final_real[:, -1, :]
    last_state_imag = final_imag[:, -1, :]
    
    return final_real, final_imag, last_state_real, last_state_imag

# ==========================================
# 4. MODEL LAYERS
# ==========================================

class GatedSpectralConv(nn.Module):
    def __init__(self, d_model, n_frequencies):
        super().__init__()
        self.d_model = d_model
        self.n_frequencies = n_frequencies
        
        # Projections to spectral domain
        self.to_spectral = nn.Linear(d_model, n_frequencies * 2, bias=False)
        self.x_to_decay = nn.Linear(d_model, n_frequencies)
        
        # Learnable frequencies
        self.frequencies = nn.Parameter(torch.randn(n_frequencies))
        
        # Projection back
        self.from_spectral = nn.Linear(n_frequencies * 2, d_model, bias=False)
        self.register_buffer('dt', torch.tensor(1.0))

    def forward(self, x, state=None):
        batch_size, seq_len, _ = x.shape
        
        # Calculate decay (forget gate)
        decay_logits = self.x_to_decay(x)
        decay = torch.sigmoid(decay_logits) 

        # Calculate rotation (oscillation)
        omega = self.frequencies * 0.1 * self.dt
        rot_real = torch.cos(omega)
        rot_imag = torch.sin(omega)
        
        # Project input to complex space
        spectral = self.to_spectral(x)
        u_real = spectral[..., :self.n_frequencies]
        u_imag = spectral[..., self.n_frequencies:]

        # Initialize state if needed
        if state is None:
            state_real = torch.zeros(batch_size, self.n_frequencies, device=x.device, dtype=x.dtype)
            state_imag = torch.zeros(batch_size, self.n_frequencies, device=x.device, dtype=x.dtype)
        else:
            state_real, state_imag = state
            
        # Run Recurrence
        y_real, y_imag, new_state_real, new_state_imag = parallel_gated_recurrence(
            u_real, u_imag, decay, rot_real, rot_imag, state_real, state_imag
        )
        
        # Project back
        y = torch.cat([y_real, y_imag], dim=-1)
        return self.from_spectral(y), (new_state_real, new_state_imag)

class SpectralBlock(nn.Module):
    def __init__(self, d_model, n_frequencies, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = GatedSpectralConv(d_model, n_frequencies)
        
        # Gating mechanisms
        self.gate = nn.Linear(d_model, d_model)
        self.out_gate = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, state=None):
        residual = x
        x = self.norm(x)
        
        # Pre-gate
        u = x * torch.sigmoid(self.gate(x))
        
        # Spectral Convolution
        y, new_state = self.conv(u, state=state)
        
        # Post-gate
        y = y * torch.sigmoid(self.out_gate(x))
        y = self.dropout(y)
        
        # Residual + FFN
        x = residual + y
        x = x + self.ffn(x)
        return x, new_state

# ==========================================
# 5. THE NEW COMPONENT: RESIDUAL VIB
# ==========================================

class ResidualVIB(nn.Module):
    """
    Residual Variational Information Bottleneck.
    Acts as a 'Concept Co-Processor'.
    """
    def __init__(self, input_dim, latent_dim, dropout=0.1):
        super().__init__()
        # Encoder: Compresses input to latent distribution parameters
        self.encoder = nn.Linear(input_dim, latent_dim * 2) 
        
        # Decoder: Expands latent sample back to residual update
        self.decoder = nn.Linear(latent_dim, input_dim)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # x shape: [Batch, Seq, Dim]
        
        # 1. Encode
        # We use LayerNorm before encoding to stabilize the latent space
        x_norm = self.layer_norm(x)
        params = self.encoder(x_norm)
        mu, logvar = params.chunk(2, dim=-1)
        
        # 2. Sample
        z = self.reparameterize(mu, logvar)
        
        # 3. Decode
        # This output is the "Concept Residual"
        out = self.decoder(z)
        out = self.dropout(self.activation(out))
        
        return out, mu, logvar

# ==========================================
# 6. MAIN MODEL
# ==========================================

class SpectralGatingNetwork(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_frequencies=128,
                 latent_dim=64, dropout=0.1, kl_weight=0.01):
        super().__init__()
        self.d_model = d_model
        self.kl_weight = kl_weight
        self.kl_step = 0
        self.kl_anneal_steps = 5000 

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # The Deterministic Stack (Syntax/Details)
        self.layers = nn.ModuleList([
            SpectralBlock(d_model, n_frequencies, dropout) for _ in range(n_layers)
        ])
        
        # The Variational Path (Concepts/Gist)
        # We place this parallel to the output projection
        self.res_vib = ResidualVIB(d_model, latent_dim, dropout)

        # Final Output
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight 

    def update_regularization(self, p):
        """Watcher calls this to update dropout dynamically"""
        self.dropout.p = p
        self.res_vib.dropout.p = p
        for layer in self.layers:
            layer.dropout.p = p
            layer.ffn[4].p = p 

    def get_kl_weight(self):
        # Simple linear annealing
        if not self.training: return self.kl_weight
        progress = min(1.0, self.kl_step / self.kl_anneal_steps)
        return self.kl_weight * progress

    def forward(self, x, states=None):
        B, T = x.shape
        
        # 1. Embedding
        x = self.token_emb(x)
        x = self.dropout(x)

        # 2. Deterministic Path (Spectral Layers)
        # This preserves high-frequency details (grammar, specific names)
        new_states = []
        deterministic_signal = x
        for i, layer in enumerate(self.layers):
            current_state = states[i] if states else None
            deterministic_signal, state = layer(deterministic_signal, state=current_state)
            new_states.append(state)

        # 3. Variational Path (Residual VIB)
        # Input: The processed deterministic signal (so it summarizes context)
        # Output: A "Concept" vector to be added back
        concept_signal, mu, logvar = self.res_vib(deterministic_signal)
        
        # 4. The Merge (Bicameral Integration)
        # Output = Syntax + Concept
        combined_signal = deterministic_signal + concept_signal

        # 5. Projection
        out = self.norm_f(combined_signal)
        logits = self.lm_head(out)

        # 6. Loss Calculation
        # KL Divergence with 'Free Bits' (min_kl) to prevent collapse
        kl_raw = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_mean = torch.mean(kl_raw) # Mean over batch and seq
        min_kl = 0.05 
        kl_loss_clamped = torch.max(kl_mean, torch.tensor(min_kl, device=x.device))
        reg_loss = self.get_kl_weight() * kl_loss_clamped
        
        if self.training:
            self.kl_step += 1

        return logits, reg_loss, new_states

# ==========================================
# 7. TRAINING & INFERENCE
# ==========================================

@torch.no_grad()
def estimate_loss(model, data, eval_iters=50, batch_size=32, seq_len=64, device='cpu'):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        if len(data) <= seq_len + 1: break
        x, y = get_batch(data, batch_size, seq_len, device)
        logits, reg_loss, _ = model(x)
        loss_token = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss = loss_token + reg_loss
        losses.append(total_loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="The", max_new_tokens=100, top_k=50):
    model.eval()
    tokens = tokenizer.encode(prompt)
    context = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Initialize states for generation
    states = None
    
    for _ in range(max_new_tokens):
        # Crop context if too long (though spectral layers handle long ctx well)
        cond_ctx = context[:, -512:] 
        
        # Forward pass
        logits, _, states = model(cond_ctx, states=None) # Reset states for simplicity in this snippet
        logits = logits[:, -1, :]
        
        # Sampling
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        context = torch.cat((context, next_token), dim=1)
        
    output_text = tokenizer.decode(context[0].tolist())
    model.train()
    return output_text

def train(model, data_source, tokenizer, steps=10000, batch_size=16, seq_len=128, lr=1e-4, device='cuda'):
    # Initialize Watcher
    watcher = HomeostaticRegulator(target_gap=0.15)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=watcher.current_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    
    model.train()
    
    # Split Data
    n = int(0.9 * len(data_source))
    train_data = data_source[:n]
    val_data = data_source[n:]
    
    print(f"{'Step':<6} | {'Train':<7} | {'Val':<7} | {'Gap':<6} | {'Drop%':<6} | {'WD':<7} | {'Status'}")
    print("-" * 75)
    
    step = 0
    best_val_loss = float('inf')
    
    while step < steps:
        x, y = get_batch(train_data, batch_size, seq_len, device)
        
        logits, reg_loss, _ = model(x)
        loss_token = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Total Loss = Reconstruction + KL Penalty
        loss = loss_token + reg_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        step += 1
        
        # --- THE WATCHER INTERVENES ---
        if step % 100 == 0:
            v_loss = estimate_loss(model, val_data, batch_size=batch_size, seq_len=seq_len, device=device)
            t_loss = loss.item()
            
            # 1. Ask Watcher for decision
            decision = watcher.update(t_loss, v_loss)
            
            # 2. Apply to Model
            model.update_regularization(decision['dropout'])
            
            # 3. Apply to Optimizer
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = decision['weight_decay']
            
            # 4. Log
            print(f"{step:<6} | {t_loss:.4f}  | {v_loss:.4f}  | {decision['gap']:.4f} | "
                  f"{decision['dropout']:.3f}  | {decision['weight_decay']:.5f} | {decision['status']}")
            
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(model.state_dict(), "best_model.pt")

        if step % 1000 == 0:
            print(f"\n--- Prediction at Step {step} ---")
            sample_text = generate_sample(model, tokenizer, device, prompt="The future of AI is", top_k=10)
            print(f"Generated: {sample_text}\n-------------------------------")
    
    print("Training finished.")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup Tokenizer
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    except:
        print("Could not load GPT2 tokenizer. Please install transformers.")
        exit()
    
    # 2. Setup Model
    # Note: latent_dim is small (32) compared to d_model (384) to force abstraction
    model = SpectralGatingNetwork(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        n_layers=6,
        n_frequencies=256,
        latent_dim=32,       # Small latent space for "Concepts"
        dropout=0.1,         # Watcher will adjust this
        kl_weight=0.05
    ).to(device)

    # 3. Load Data (Dummy data for demonstration)
    print("Generating dummy data for training...")
    if not os.path.exists("input.txt"):
        with open("input.txt", "w") as f:
            f.write("The quick brown fox jumps over the lazy dog. " * 5000)
    
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    #if len(text) > 500000: text = text[:500000]
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Train
    train(model, data, tokenizer, steps=10000, batch_size=6, seq_len=512, device=device)