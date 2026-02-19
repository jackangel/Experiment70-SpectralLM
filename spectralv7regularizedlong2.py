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
        if not self.parquet_files: 
            print(f"No parquet files found in {folder_path}.")
        else:
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
# 2. JIT-Compiled Gated Recurrence (FIXED BROADCASTING)
# -----------------------------
@torch.jit.script
def gated_recurrence_scan(u_real: torch.Tensor, u_imag: torch.Tensor, 
                         decay: torch.Tensor, 
                         rot_real: torch.Tensor, rot_imag: torch.Tensor,
                         state_real: torch.Tensor, state_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    u: (Batch, Seq_Len, Dim)
    decay: (Batch, Seq_Len, Dim)
    rot: (Dim) -> Needs to be broadcast to (Batch, Dim)
    state: (Batch, Dim)
    """
    B, L, D = u_real.shape
    
    # Pre-allocate output lists
    out_real = torch.jit.annotate(List[torch.Tensor], [])
    out_imag = torch.jit.annotate(List[torch.Tensor], [])
    
    current_real = state_real
    current_imag = state_imag
    
    # Expand rotation to match batch size for easier broadcasting if needed,
    # though standard broadcasting usually handles (B, D) * (D).
    # We ensure strict shapes here.
    
    for t in range(L):
        # 1. Apply Decay
        # decay[:, t, :] is (B, D)
        d = decay[:, t, :]
        
        s_real = current_real * d
        s_imag = current_imag * d
        
        # 2. Apply Rotation
        # rot_real is (D), s_real is (B, D). Broadcast works automatically.
        r_real = s_real * rot_real - s_imag * rot_imag
        r_imag = s_real * rot_imag + s_imag * rot_real
        
        # 3. Add Input
        current_real = r_real + u_real[:, t, :]
        current_imag = r_imag + u_imag[:, t, :]
        
        out_real.append(current_real)
        out_imag.append(current_imag)
        
    stacked_real = torch.stack(out_real, dim=1)
    stacked_imag = torch.stack(out_imag, dim=1)
    
    return stacked_real, stacked_imag, current_real, current_imag

# -----------------------------
# 3. Gated Spectral Components
# -----------------------------
class GatedSpectralConv(nn.Module):
    def __init__(self, d_model, n_frequencies):
        super().__init__()
        self.d_model = d_model
        self.n_frequencies = n_frequencies
        
        self.to_spectral = nn.Linear(d_model, n_frequencies * 2, bias=False)
        self.x_to_decay = nn.Linear(d_model, n_frequencies)
        self.frequencies = nn.Parameter(torch.randn(n_frequencies))
        self.from_spectral = nn.Linear(n_frequencies * 2, d_model, bias=False)
        self.register_buffer('dt', torch.tensor(1.0))

    def forward(self, x, state=None):
        batch_size, seq_len, _ = x.shape
        
        # A. Compute Decay
        decay_logits = self.x_to_decay(x)
        decay = torch.sigmoid(decay_logits) 

        # B. Compute Rotation
        omega = self.frequencies * 0.1 * self.dt
        rot_real = torch.cos(omega)
        rot_imag = torch.sin(omega)
        
        # C. Prepare Input
        spectral = self.to_spectral(x)
        u_real = spectral[..., :self.n_frequencies]
        u_imag = spectral[..., self.n_frequencies:]

        # D. Initialize State
        if state is None:
            state_real = torch.zeros(batch_size, self.n_frequencies, device=x.device, dtype=x.dtype)
            state_imag = torch.zeros(batch_size, self.n_frequencies, device=x.device, dtype=x.dtype)
        else:
            state_real, state_imag = state
            
        # E. Run Recurrence
        y_real, y_imag, new_state_real, new_state_imag = gated_recurrence_scan(
            u_real, u_imag, decay, rot_real, rot_imag, state_real, state_imag
        )
        
        # F. Project back
        y = torch.cat([y_real, y_imag], dim=-1)
        return self.from_spectral(y), (new_state_real, new_state_imag)

class SpectralBlock(nn.Module):
    def __init__(self, d_model, n_frequencies, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = GatedSpectralConv(d_model, n_frequencies)
        
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
        # This method handles the logic. 
        # If checkpointing is used, we only checkpoint the heavy parts, not the state passing.
        
        residual = x
        x = self.norm(x)
        
        # Pre-conv gating
        u = x * torch.sigmoid(self.gate(x))
        
        if conditioning is not None:
            u = u + conditioning
            
        # Gated Recurrence
        # We cannot easily checkpoint this specific call if 'state' is involved 
        # because 'state' changes size/content.
        y, new_state = self.conv(u, state=state)
        
        # Post-conv gating
        y = y * torch.sigmoid(self.out_gate(x))
        y = self.dropout(y)
        
        x = residual + y
        x = x + self.ffn(x)
        return x, new_state

# -----------------------------
# 4. Stabilized Variational Bottleneck
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
# 5. Language Model
# -----------------------------
class SpectralBottleneckLM(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_frequencies=128,
                 bottleneck_dim=192, dropout=0.1, kl_weight=0.01):
        super().__init__()
        self.d_model = d_model
        self.kl_weight = kl_weight
        self.kl_step = 0
        self.kl_anneal_steps = 10000 

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
        
        self.use_checkpointing = False 

    def get_kl_weight(self):
        if not self.training: return self.kl_weight
        progress = min(1.0, self.kl_step / self.kl_anneal_steps)
        return self.kl_weight * progress

    def forward(self, x, states=None):
        B, T = x.shape

        h = self.token_emb(x)
        h = self.dropout(h)

        bottleneck_out, mu, logvar = self.bottleneck(h)
        h = h + bottleneck_out 

        new_states = []
        for i, layer in enumerate(self.token_layers):
            cond = self.cross_projections[i](bottleneck_out)
            current_state = states[i] if states else None
            
            # FIXED CHECKPOINTING LOGIC
            # We only checkpoint the stateless part if possible, or we skip checkpointing 
            # for the recurrent layer to avoid metadata mismatch errors.
            # Given the complexity of state passing, we will disable checkpointing 
            # for the recurrent block wrapper to ensure stability, or apply it carefully.
            
            # To fix the error: Run the layer normally. The memory savings from checkpointing 
            # a simple RNN scan are minimal compared to the headache of state management.
            if self.use_checkpointing and current_state is None:
                # Only checkpoint if we don't have a carry-over state (e.g. initial training)
                # But even then, 'layer' returns a tuple (h, state), which checkpoint handles poorly
                # if the shapes aren't static tensors.
                # STRATEGY: Run normally.
                h, state = layer(h, conditioning=cond, state=current_state)
            else:
                h, state = layer(h, conditioning=cond, state=current_state)
            
            new_states.append(state)

        h = self.ln_f(h)
        logits = self.lm_head(h)

        # KL Loss
        kl_raw = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_mean = torch.mean(kl_raw)
        min_kl = 0.05
        kl_loss_clamped = torch.max(kl_mean, torch.tensor(min_kl, device=x.device))
        reg_loss = self.get_kl_weight() * kl_loss_clamped
        
        if self.training:
            self.kl_step += 1

        return logits, reg_loss, new_states

# -----------------------------
# 6. Training Logic
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
        logits, reg_loss, _ = model(x)
        loss_token = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss = loss_token + reg_loss
        losses.append(total_loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="The", max_new_tokens=200, top_k=50):
    model.eval()
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = context
    
    for _ in range(max_new_tokens):
        cond_ctx = generated[:, -512:] 
        logits, _, _ = model(cond_ctx)
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
    # Disable checkpointing to prevent metadata errors during recurrence
    model.use_checkpointing = False 
    print(f"Gradient Checkpointing Enabled: {model.use_checkpointing}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    
    if isinstance(data_source, torch.Tensor):
        n = int(0.9 * len(data_source))
        train_data = data_source[:n]
        val_data = data_source[n:]
        is_tensor = True
    else:
        is_tensor = False
        val_data = None 
    
    print(f"Starting Training with Gated Spectral Conv & Annealed KL...")
    
    step = 0
    while step < steps:
        if is_tensor:
            x, y = get_batch(train_data, batch_size, seq_len, device)
            batches = [(x, y)]
        else:
            batches = [] 
            
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
                kl_w = model.get_kl_weight()
                print(f"Step {step} | Loss: {loss.item():.4f} (Tok: {loss_token:.3f}, KL: {reg_loss:.3f}) | Val: {v_loss:.4f} | KL_W: {kl_w:.5f}")

            if step % 1000 == 0:
                print(f"\n--- Prediction at Step {step} ---")
                sample_text = generate_sample(model, tokenizer, device, prompt="The", top_k=10)
                print(f"Generated: {sample_text}\n-------------------------------")
            
            if step >= steps: break
    
    print("Training finished. Saving checkpoint...")
    torch.save(model.state_dict(), "checkpoint_gated.pt")
    print("Checkpoint saved to checkpoint_gated.pt")

# -----------------------------
# 7. Infinite Inference Mode
# -----------------------------
@torch.no_grad()
def chat_infinite(model, tokenizer, device):
    model.eval()
    model.use_checkpointing = False
    states = None 
    
    print("\n--- Infinite Context Chat (Gated Recurrent Mode) ---")
    print("Type 'exit' to quit. Type 'reset' to clear memory.")
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]: break
            if user_input.lower() == "reset":
                states = None
                print("Memory reset.")
                continue

            context = torch.tensor([tokenizer.encode(user_input)], dtype=torch.long, device=device)
            logits, _, states = model(context, states=states)
            
            print("Bot: ", end="", flush=True)
            
            curr_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            for _ in range(500): 
                logits, _, states = model(curr_token, states=states)
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
    
    model = SpectralBottleneckLM(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        n_layers=6,
        n_frequencies=256,
        bottleneck_dim=128,
        dropout=0.25, 
        kl_weight=0.01 
    ).to(device)

    if os.path.exists("checkpoint_gated.pt"):
        print("Checkpoint found. Loading...")
        model.load_state_dict(torch.load("checkpoint_gated.pt", map_location=device))
        chat_infinite(model, tokenizer, device)
    else:
        print("No checkpoint found. Generating dummy data for training...")
        if not os.path.exists("input.txt"):
            with open("input.txt", "w") as f:
                f.write("The quick brown fox jumps over the lazy dog. " * 1000)
        
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        
        if len(text) > 1000000: text = text[:1000000]
            
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")
        
        train(model, data, tokenizer, steps=100000, batch_size=6, seq_len=256, device=device)