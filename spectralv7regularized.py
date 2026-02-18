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

# -----------------------------
# 1. Data Loader (unchanged)
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
# 2. Spectral Components (unchanged)
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

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        spectral = self.to_spectral(x)
        real = spectral[..., :self.n_frequencies]
        imag = spectral[..., self.n_frequencies:]
        
        u = torch.complex(real, imag)
        decay = torch.sigmoid(self.log_decay)
        omega = self.frequencies * 0.1 * self.dt
        rot = torch.complex(torch.cos(omega), torch.sin(omega))
        
        # Construct Kernel
        A = decay * rot
        t = torch.arange(seq_len, device=device).float().unsqueeze(-1)
        A_expanded = A.view(1, 1, self.n_frequencies).expand(1, seq_len, self.n_frequencies)
        
        ones = torch.ones(1, 1, self.n_frequencies, device=device, dtype=A.dtype)
        powers = torch.cumprod(torch.cat([ones, A_expanded[:, :-1, :]], dim=1), dim=1)
        kernel = powers * rot.view(1, 1, self.n_frequencies)
        
        # FFT Convolution
        n_fft = 2 * seq_len
        U_f = torch.fft.fft(u, n=n_fft, dim=1)
        K_f = torch.fft.fft(kernel, n=n_fft, dim=1)
        y = torch.fft.ifft(U_f * K_f, n=n_fft, dim=1)[:, :seq_len, :]
        
        spectral_out = torch.cat([y.real, y.imag], dim=-1)
        return self.from_spectral(spectral_out)

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

    def forward(self, x, conditioning=None):
        residual = x
        x = self.norm(x)
        
        u = x * torch.sigmoid(self.gate(x))
        if conditioning is not None:
            u = u + conditioning
            
        y = self.conv(u)
        y = y * torch.sigmoid(self.out_gate(x))
        y = self.dropout(y)
        
        x = residual + y
        x = x + self.ffn(x)
        return x

# -----------------------------
# 3. Simple Bottleneck Module (new)
# -----------------------------
class SimpleBottleneck(nn.Module):
    """
    Compresses token embeddings to a lower dimension, applies regularization
    (noise + dropout), and projects back to original dimension.
    """
    def __init__(self, d_model, bottleneck_dim, dropout=0.1, noise_std=0.1):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.noise_std = noise_std

    def forward(self, x):
        # x: (B, T, D)
        z = self.down(x)
        if self.training and self.noise_std > 0:
            z = z + torch.randn_like(z) * self.noise_std
        z = self.dropout(z)
        out = self.up(z)
        return out, z  # return both output and bottleneck activations

# -----------------------------
# 4. Simplified Language Model with Bottleneck (replaces SpectralBucketLM)
# -----------------------------
class SpectralBottleneckLM(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_frequencies=128,
                 bottleneck_dim=192, dropout=0.1, noise_std=0.1, reg_weight=0.01):
        super().__init__()
        self.d_model = d_model
        self.reg_weight = reg_weight

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # Bottleneck
        self.bottleneck = SimpleBottleneck(d_model, bottleneck_dim, dropout, noise_std)

        # Token stream layers
        self.token_layers = nn.ModuleList([
            SpectralBlock(d_model, n_frequencies, dropout) for _ in range(n_layers)
        ])

        # Cross projections: each layer gets a projection of the bottleneck output
        self.cross_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(n_layers)
        ])

        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

    def forward(self, x):
        B, T = x.shape

        # 1. Token embeddings
        h = self.token_emb(x)
        h = self.dropout(h)

        # 2. Bottleneck: compress, regularize, reconstruct
        bottleneck_out, z = self.bottleneck(h)  # (B, T, D) and (B, T, bottleneck_dim)

        # 3. Inject bottleneck into token stream (initial addition + per‑layer conditioning)
        h = h + bottleneck_out  # initial injection

        for i, layer in enumerate(self.token_layers):
            cond = self.cross_projections[i](bottleneck_out)  # (B, T, D)
            h = layer(h, conditioning=cond)

        # 4. Final prediction
        h = self.ln_f(h)
        logits = self.lm_head(h)

        # 5. Regularization loss (L2 penalty on bottleneck activations)
        reg_loss = self.reg_weight * torch.mean(z ** 2)

        return logits, reg_loss

# -----------------------------
# 5. Training Logic (adapted)
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
        logits, reg_loss = model(x)
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
        logits, _ = model(generated)
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
    
    print(f"Starting Training with Bottleneck (dim={model.bottleneck.down.out_features})...")
    
    step = 0
    while step < steps:
        if is_tensor:
            x, y = get_batch(train_data, batch_size, seq_len, device)
            batches = [(x, y)]
        else:
            # If using parquet loader, iterate here (not implemented for simplicity)
            pass 
            
        for x, y in batches:
            logits, reg_loss = model(x)
            loss_token = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss_token + reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step += 1
            if step % 100 == 0:
                v_loss = estimate_loss(model, val_data, batch_size=batch_size, seq_len=seq_len, device=device) if is_tensor else 0.0
                print(f"Step {step} | Loss: {loss.item():.4f} (Tok: {loss_token:.3f}, Reg: {reg_loss:.3f}) | Val: {v_loss:.4f}")

            if step % 1000 == 0:
                print(f"\n--- Prediction at Step {step} ---")
                sample_text = generate_sample(model, tokenizer, device, prompt="The", top_k=10)
                print(f"Generated: {sample_text}\n-------------------------------")
            
            if step >= steps: break
    
    print("Training finished. Saving checkpoint...")
    torch.save(model.state_dict(), "checkpoint_bottleneck.pt")
    print("Checkpoint saved to checkpoint_bottleneck.pt")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Initialize the simplified bottleneck model
    model = SpectralBottleneckLM(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        n_layers=6,
        n_frequencies=256,
        bottleneck_dim=128,      # compression factor 3x (384→128)
        dropout=0.2,
        noise_std=0.1,
        reg_weight=0.01
    ).to(device)

    # Load checkpoint if exists (adjusted for new model)
    if os.path.exists("checkpoint_bottleneck.pt"):
        print("Checkpoint found. Loading and entering Chat Mode...")
        model.load_state_dict(torch.load("checkpoint_bottleneck.pt", map_location=device))
        model.eval()
        
        print("\n--- Chat Mode ---")
        print("Type 'exit' to quit.")
        
        while True:
            try:
                user_input = input("\nUser: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                # Settings
                try:
                    temp_str = input("Temperature (default 1.0): ")
                    temperature = float(temp_str) if temp_str.strip() else 1.0
                    
                    top_k_str = input("Top K (default 50): ")
                    top_k = int(top_k_str) if top_k_str.strip() else 50
                    
                    top_p_str = input("Top P (default 0.9): ")
                    top_p = float(top_p_str) if top_p_str.strip() else 0.9
                except ValueError:
                    print("Invalid setting, using defaults.")
                    temperature = 1.0
                    top_k = 50
                    top_p = 0.9

                context = torch.tensor([tokenizer.encode(user_input)], dtype=torch.long, device=device)
                generated = context
                
                print("Bot: ", end="", flush=True)
                # Generate up to 200 tokens
                for _ in range(200):
                    with torch.no_grad():
                        logits, _ = model(generated)
                        logits = logits[:, -1, :] / temperature
                        
                        # Top K
                        if top_k > 0:
                            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = -float('Inf')
                        
                        # Top P (Nucleus Sampling)
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            logits[indices_to_remove] = -float('Inf')

                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        generated = torch.cat((generated, next_token), dim=1)
                        
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                            
                        print(tokenizer.decode([next_token.item()]), end="", flush=True)
                print()
                
            except KeyboardInterrupt:
                print("\nExiting Chat Mode.")
                break
    else:
        # Prepare dummy data
        if not os.path.exists("input.txt"):
            with open("input.txt", "w") as f:
                f.write("The quick brown fox jumps over the lazy dog. " * 1000)
                
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")
        
        train(model, data, tokenizer, steps=100000, batch_size=6, seq_len=512, device=device)
        
        print("\nFinal Generation Test:")
        context = torch.tensor([tokenizer.encode("The quick")], dtype=torch.long, device=device)
        model.eval()
        for _ in range(200):
            logits, _ = model(context)
            next_token = torch.argmax(logits[0, -1]).item()
            context = torch.cat([context, torch.tensor([[next_token]], device=device)], dim=1)
            print(tokenizer.decode([next_token]), end="", flush=True)
        print()