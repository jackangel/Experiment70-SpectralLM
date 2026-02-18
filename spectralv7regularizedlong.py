import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import glob
import random
from pathlib import Path
from transformers import GPT2TokenizerFast
import pyarrow.parquet as pq
from torch.utils.checkpoint import checkpoint 

# -----------------------------
# 1. Data Loader
# -----------------------------
class ParquetDataLoader:
    def __init__(self, folder_path, text_column, tokenizer, chunk_size=10000):
        self.folder_path = Path(folder_path)
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.parquet_files = sorted(glob.glob(str(self.folder_path / "**" / "*.parquet"), recursive=True))
        if not self.parquet_files: raise ValueError(f"No parquet files found in {folder_path}")
        print(f"Found {len(self.parquet_files)} parquet files")
    
    def load_file_in_chunks(self, file_path):
        try:
            parquet_file = pq.ParquetFile(file_path)
            all_tokens = []
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                df = batch.to_pandas()
                if self.text_column not in df.columns: continue
                for text in df[self.text_column]:
                    if isinstance(text, str) and text.strip():
                        all_tokens.extend(self.tokenizer.encode(text))
                if len(all_tokens) > 1000000:
                    yield torch.tensor(all_tokens, dtype=torch.long)
                    all_tokens = []
            if all_tokens: yield torch.tensor(all_tokens, dtype=torch.long)
        except Exception as e: print(f"Error reading {file_path}: {e}")
    
    def iterate_files(self):
        for file_path in self.parquet_files:
            yield file_path, self.load_file_in_chunks(file_path)

# -----------------------------
# 2. Spectral Components (Enhanced for Infinite Inference)
# -----------------------------
class SpectralConv(nn.Module):
    def __init__(self, d_model, n_frequencies):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.to_spectral = nn.Linear(d_model, n_frequencies * 2, bias=False)
        self.log_decay = nn.Parameter(torch.linspace(-3.0, -0.001, n_frequencies)) 
        self.frequencies = nn.Parameter(torch.randn(n_frequencies))
        self.from_spectral = nn.Linear(n_frequencies * 2, d_model, bias=False)
        self.register_buffer('dt', torch.tensor(1.0))

    def get_system_matrices(self):
        """Precompute the recurrence matrices A and rot."""
        decay = torch.sigmoid(self.log_decay)
        omega = self.frequencies * 0.1 * self.dt
        rot = torch.complex(torch.cos(omega), torch.sin(omega))
        A = decay * rot
        return A, rot

    def forward(self, x, state=None):
        """Supports both FFT Parallel mode and Recurrent Step mode."""
        batch_size, seq_len, _ = x.shape
        A, rot = self.get_system_matrices()
        
        spectral = self.to_spectral(x)
        u = torch.complex(spectral[..., :self.n_frequencies], spectral[..., self.n_frequencies:])

        if seq_len > 1:
            # --- FFT Parallel Mode (Training) ---
            n_fft = 2 * seq_len
            t = torch.arange(seq_len, device=x.device).float().unsqueeze(-1)
            A_expanded = A.view(1, 1, self.n_frequencies).expand(1, seq_len, self.n_frequencies)
            ones = torch.ones(1, 1, self.n_frequencies, device=x.device, dtype=A.dtype)
            powers = torch.cumprod(torch.cat([ones, A_expanded[:, :-1, :]], dim=1), dim=1)
            kernel = powers * rot.view(1, 1, self.n_frequencies)
            
            U_f = torch.fft.fft(u, n=n_fft, dim=1)
            K_f = torch.fft.fft(kernel, n=n_fft, dim=1)
            y = torch.fft.ifft(U_f * K_f, n=n_fft, dim=1)[:, :seq_len, :]
            return self.from_spectral(torch.cat([y.real, y.imag], dim=-1)), None
        else:
            # --- Recurrent Mode (Infinite Inference) ---
            # h_t = A * h_{t-1} + u_t
            if state is None:
                state = torch.zeros(batch_size, 1, self.n_frequencies, device=x.device, dtype=u.dtype)
            
            new_state = A * state + u
            spectral_out = torch.cat([new_state.real, new_state.imag], dim=-1)
            return self.from_spectral(spectral_out), new_state

class SpectralBlock(nn.Module):
    def __init__(self, d_model, n_frequencies, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = SpectralConv(d_model, n_frequencies)
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

    def forward(self, x, conditioning=None, state=None):
        residual = x
        x = self.norm(x)
        u = x * torch.sigmoid(self.gate(x))
        if conditioning is not None:
            u = u + conditioning
            
        y, new_state = self.conv(u, state=state)
        y = y * torch.sigmoid(self.out_gate(x))
        y = self.dropout(y)
        
        x = residual + y
        x = x + self.ffn(x)
        return x, new_state

# -----------------------------
# 3. Variational Bottleneck Module
# -----------------------------
class VariationalBottleneck(nn.Module):
    def __init__(self, d_model, bottleneck_dim, dropout=0.1):
        super().__init__()
        self.fc_mu = nn.Linear(d_model, bottleneck_dim)
        self.fc_logvar = nn.Linear(d_model, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu 
            
        z = self.dropout(z)
        out = self.up(z)
        return out, mu, logvar

# -----------------------------
# 4. Language Model (Enhanced with Checkpointing & States)
# -----------------------------
class SpectralBottleneckLM(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_frequencies=128,
                 bottleneck_dim=192, dropout=0.1, kl_weight=0.01):
        super().__init__()
        self.d_model = d_model
        self.kl_weight = kl_weight

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.bottleneck = VariationalBottleneck(d_model, bottleneck_dim, dropout)

        self.token_layers = nn.ModuleList([
            SpectralBlock(d_model, n_frequencies, dropout) for _ in range(n_layers)
        ])

        self.cross_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight 
        
        # Flag to enable gradient checkpointing for large contexts
        self.use_checkpointing = False 

    def forward(self, x, states=None):
        B, T = x.shape

        h = self.token_emb(x)
        h = self.dropout(h)

        # Variational forward pass
        bottleneck_out, mu, logvar = self.bottleneck(h)
        h = h + bottleneck_out 

        new_states = []
        for i, layer in enumerate(self.token_layers):
            cond = self.cross_projections[i](bottleneck_out)
            current_state = states[i] if states else None
            
            # VRAM Optimization: Gradient Checkpointing
            if self.training and self.use_checkpointing:
                # Note: Checkpointing requires inputs to require_grad for autograd to work, 
                # but h usually does. We pass use_reentrant=False for modern PyTorch.
                h, state = checkpoint(layer, h, cond, current_state, use_reentrant=False)
            else:
                h, state = layer(h, conditioning=cond, state=current_state)
            
            new_states.append(state)

        h = self.ln_f(h)
        logits = self.lm_head(h)

        # KL Divergence Loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        reg_loss = self.kl_weight * kl_loss

        return logits, reg_loss, new_states

# -----------------------------
# 5. Training Logic
# -----------------------------
def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, eval_iters=50, batch_size=32, seq_len=64, device='cpu'):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        if len(data) <= seq_len + 1: break
        x, y = get_batch(data, batch_size, seq_len, device)
        logits, reg_loss, _ = model(x) # Ignore states during batch training
        loss_token = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss = loss_token + reg_loss
        losses.append(total_loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="The", max_new_tokens=200, top_k=50):
    """Legacy generation (non-recurrent) for quick sanity checks during training."""
    model.eval()
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = context
    for _ in range(max_new_tokens):
        logits, _, _ = model(generated)
        logits = logits[:, -1, :]
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
    output_text = tokenizer.decode(generated[0].tolist())
    model.train()
    return output_text

def train(model, data_source, tokenizer, steps=100000, batch_size=8, seq_len=128, lr=3e-4, device='cuda'):
    # Enable checkpointing for training to save VRAM
    model.use_checkpointing = True 
    print(f"Gradient Checkpointing Enabled: {model.use_checkpointing}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    if isinstance(data_source, torch.Tensor):
        n = int(0.9 * len(data_source))
        train_data = data_source[:n]
        val_data = data_source[n:]
        is_tensor = True
    else:
        is_tensor = False
        val_data = None 
    
    print(f"Starting Training with Variational Bottleneck (KL Weight={model.kl_weight})...")
    
    step = 0
    while step < steps:
        if is_tensor:
            x, y = get_batch(train_data, batch_size, seq_len, device)
            batches = [(x, y)]
        else:
            pass 
            
        for x, y in batches:
            logits, reg_loss, _ = model(x)
            loss_token = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss_token + reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step += 1
            if step % 100 == 0:
                v_loss = estimate_loss(model, val_data, batch_size=batch_size, seq_len=seq_len, device=device) if is_tensor else 0.0
                print(f"Step {step} | Loss: {loss.item():.4f} (Tok: {loss_token:.3f}, KL: {reg_loss:.3f}) | Val: {v_loss:.4f}")

            if step % 1000 == 0:
                print(f"\n--- Prediction at Step {step} ---")
                sample_text = generate_sample(model, tokenizer, device, prompt="The", top_k=10)
                print(f"Generated: {sample_text}\n-------------------------------")
            
            if step >= steps: break
    
    print("Training finished. Saving checkpoint...")
    torch.save(model.state_dict(), "checkpoint_bottleneck.pt")
    print("Checkpoint saved to checkpoint_bottleneck.pt")

# -----------------------------
# 6. Infinite Inference Mode
# -----------------------------
@torch.no_grad()
def chat_infinite(model, tokenizer, device):
    model.eval()
    model.use_checkpointing = False # Not needed for inference
    states = None 
    
    print("\n--- Infinite Context Chat (Recurrent Mode) ---")
    print("Type 'exit' to quit. Type 'reset' to clear memory.")
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]: break
            if user_input.lower() == "reset":
                states = None
                print("Memory reset.")
                continue

            # 1. Process User Input
            # We process the user prompt in parallel (FFT mode) if it's long, 
            # or sequentially if we want to be strict. 
            # For simplicity here, we feed the prompt.
            context = torch.tensor([tokenizer.encode(user_input)], dtype=torch.long, device=device)
            
            # If we have no history, we process the whole prompt.
            # If we have history (states), we process the prompt *given* that history.
            logits, _, states = model(context, states=states)
            
            # 2. Generate Bot Response (Token by Token)
            print("Bot: ", end="", flush=True)
            
            # Start generation from the last token of the prompt
            curr_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            # Generation loop
            for _ in range(500): # Max response length
                # Forward pass for ONE token using the recurrent state
                # This is O(1) complexity regardless of history length
                logits, _, states = model(curr_token, states=states)
                
                # Sampling
                next_token_logits = logits[:, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1)
                
                word = tokenizer.decode([token_id.item()])
                print(word, end="", flush=True)
                
                curr_token = token_id
                if token_id.item() == tokenizer.eos_token_id:
                    break
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Configuration
    model = SpectralBottleneckLM(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        n_layers=6,
        n_frequencies=256,
        bottleneck_dim=128,
        dropout=0.2,
        kl_weight=0.005 
    ).to(device)

    if os.path.exists("checkpoint_bottleneck.pt"):
        print("Checkpoint found. Loading...")
        model.load_state_dict(torch.load("checkpoint_bottleneck.pt", map_location=device))
        
        # Enter Infinite Chat Mode
        chat_infinite(model, tokenizer, device)
    else:
        print("No checkpoint found. Generating dummy data for training...")
        if not os.path.exists("input.txt"):
            with open("input.txt", "w") as f:
                f.write("The quick brown fox jumps over the lazy dog. " * 1000)
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")
        
        # Note: With gradient checkpointing, you can increase seq_len significantly
        # even on consumer hardware.
        train(model, data, tokenizer, steps=100000, batch_size=4, seq_len=1280, device=device)