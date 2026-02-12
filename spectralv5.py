import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import glob
import time
import random
from pathlib import Path
from transformers import GPT2TokenizerFast
import pyarrow.parquet as pq

# -----------------------------
# Parquet Data Loader
# -----------------------------
class ParquetDataLoader:
    def __init__(self, folder_path, text_column, tokenizer, chunk_size=10000):
        self.folder_path = Path(folder_path)
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        
        self.parquet_files = sorted(glob.glob(str(self.folder_path / "**" / "*.parquet"), recursive=True))
        
        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {folder_path}")
        
        print(f"Found {len(self.parquet_files)} parquet files")
    
    def load_file_in_chunks(self, file_path):
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        
        print(f"\nLoading: {Path(file_path).name}")
        
        all_tokens = []
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            df = batch.to_pandas()
            if self.text_column not in df.columns:
                continue
            
            for text in df[self.text_column]:
                if isinstance(text, str) and text.strip():
                    tokens = self.tokenizer.encode(text)
                    all_tokens.extend(tokens)
            
            if len(all_tokens) > 1000000:
                yield torch.tensor(all_tokens, dtype=torch.long)
                all_tokens = []
        
        if all_tokens:
            yield torch.tensor(all_tokens, dtype=torch.long)
    
    def get_file_count(self):
        return len(self.parquet_files)
    
    def iterate_files(self):
        for file_path in self.parquet_files:
            yield file_path, self.load_file_in_chunks(file_path)

# -----------------------------
# RoPE Implementation
# -----------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        # x: [batch, seq_len, heads, head_dim]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, None, :] # [1, seq, 1, dim]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, emb):
    # x: [batch, seq, heads, dim]
    # emb: [1, seq, 1, dim]
    return x * emb.cos() + rotate_half(x) * emb.sin()

# -----------------------------
# Causal Attention with RoPE
# -----------------------------
class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize RoPE
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        self.register_buffer('causal_mask', None)
    
    def forward(self, query, key_value=None):
        """
        If key_value is None (Layer 0), performs Self-Attention.
        If key_value is provided (Layers 1+), performs Cross-Attention to previous layer.
        """
        batch_size, seq_len, _ = query.shape
        
        # If no external KV provided, use Query as KV (Self-Attention for Layer 0)
        kv_input = key_value if key_value is not None else query
        
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            self.causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool), 
                diagonal=1
            )
        
        # Projections [Batch, Seq, Heads, Dim]
        q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(kv_input).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(kv_input).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # --- Apply RoPE (Relative Positioning) ---
        # This replaces the need for absolute positional embeddings
        freqs = self.rotary_emb(q, seq_len)
        q = apply_rotary_pos_emb(q, freqs)
        k = apply_rotary_pos_emb(k, freqs)
        # -----------------------------------------

        # Transpose for Attention: [Batch, Heads, Seq, Dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores = scores.masked_fill(self.causal_mask[:seq_len, :seq_len], float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

# -----------------------------
# Spectral State-Space Memory (S3M) Layer
# -----------------------------
class SpectralStateSpaceBlock(nn.Module):
    def __init__(self, d_model, n_frequencies, n_heads=4):
        super().__init__()
        
        self.d_model = d_model
        self.n_frequencies = n_frequencies
        
        # Matrix B (Input Projection / Gating)
        # Projects input into the complex frequency domain (Real/Imag parts)
        self.to_spectral = nn.Linear(d_model, n_frequencies * 2, bias=False)
        
        # Matrix A (Recurrence / Decay)
        # We parameterize decay to ensure it stays between 0 and 1
        # Initialized to encourage long-term memory (closer to 1.0)
        self.log_decay = nn.Parameter(torch.linspace(-2.5, -0.2, n_frequencies)) 
        self.frequencies = nn.Parameter(torch.randn(n_frequencies))
        
        # Output Projection (C)
        self.from_spectral = nn.Linear(n_frequencies * 2, d_model, bias=False)
        
        # Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Attention (Now with RoPE)
        self.attn = CausalAttention(d_model, n_heads)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )
        
        self.register_buffer('dt', torch.tensor(1.0))

    def fft_convolution(self, inputs_real, inputs_imag, decay, omega):
        """
        S3M Implementation:
        Computes the recurrence x_t = A x_{t-1} + B u_t via Convolution Theorem.
        y = IFFT( FFT(u) * FFT(K) )
        """
        batch_size, seq_len, n_freq = inputs_real.shape
        device = inputs_real.device
        
        # 1. Create the Complex Input (B, L, F)
        u = torch.complex(inputs_real, inputs_imag)
        
        # 2. Construct the Convolution Kernel (Filter)
        # Complex rotation term e^{i*omega}
        rot = torch.complex(torch.cos(omega), torch.sin(omega)) # Shape: (F)
        
        # Combined Step Multiplier: A = decay * rot
        A = decay * rot # Shape: (F)
        
        # Prepare A for cumprod broadcast (1, L, F)
        t = torch.arange(seq_len, device=device).float().unsqueeze(-1) # (L, 1)
        
        # Efficient Kernel Construction
        # We need powers of A: [A^0, A^1, A^2, ... A^{L-1}]
        # Using cumprod for speed
        A_expanded = A.view(1, 1, n_freq).expand(1, seq_len, n_freq)
        
        # Compute powers: [1, A, A^2...]
        ones = torch.ones(1, 1, n_freq, device=device, dtype=A.dtype)
        powers = torch.cumprod(torch.cat([ones, A_expanded[:, :-1, :]], dim=1), dim=1)
        
        # The Impulse Response Kernel
        # Kernel K[t] = A^t * rot
        kernel = powers * rot.view(1, 1, n_freq) # (1, L, F)
        
        # 3. Perform Convolution via FFT
        # Pad to 2L for Linear Convolution
        n_fft = 2 * seq_len
        
        U_f = torch.fft.fft(u, n=n_fft, dim=1)
        K_f = torch.fft.fft(kernel, n=n_fft, dim=1)
        
        # Multiply in frequency domain
        Y_f = U_f * K_f
        
        # Inverse FFT
        y = torch.fft.ifft(Y_f, n=n_fft, dim=1)
        
        # Crop back to original sequence length
        y = y[:, :seq_len, :]
        
        return y.real, y.imag
    
    def forward(self, x, prev_layer_output=None):
        # 1. Spectral Block (State Space Memory)
        residual = x
        x_norm = self.norm1(x)
        
        # Project to "Fixed Array" state size (n_frequencies)
        spectral = self.to_spectral(x_norm)
        real = spectral[..., :self.n_frequencies]
        imag = spectral[..., self.n_frequencies:]
        
        # Parameters for Recurrence
        decay = torch.sigmoid(self.log_decay)
        omega = self.frequencies * 0.1 * self.dt
        
        # --- FAST FFT PATH (S3M) ---
        out_real, out_imag = self.fft_convolution(real, imag, decay, omega)
        # ---------------------------
        
        spectral_out = torch.cat([out_real, out_imag], dim=-1)
        x_spectral = self.from_spectral(spectral_out)
        
        x = residual + x_spectral
        
        # 2. Attention Block (RoPE)
        residual = x
        x_norm = self.norm2(x)
        x_attn = self.attn(x_norm, prev_layer_output)
        x = residual + x_attn
        
        # 3. FFN Block
        residual = x
        x_norm = self.norm3(x)
        x_ffn = self.ffn(x_norm)
        x = residual + x_ffn
        
        return x

# -----------------------------
# Hybrid Spectral-Attention Model
# -----------------------------
class HybridSpectralLM(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_frequencies=256, n_layers=4, n_heads=4):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # REMOVED: self.pos_embedding (Replaced by RoPE in layers)
        
        self.layers = nn.ModuleList([
            SpectralStateSpaceBlock(d_model, n_frequencies, n_heads)
            for _ in range(n_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight # Weight tying
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, sequence):
        batch_size, seq_len = sequence.shape
        
        x = self.token_embedding(sequence)
        # No absolute position addition here
        
        prev_output = None
        for i, layer in enumerate(self.layers):
            # Layer 0 passes prev_output=None (Triggering Self-Attention)
            # Layers 1+ pass prev_output (Triggering Cross-Attention)
            x = layer(x, prev_output)
            prev_output = x 
        
        x = self.output_norm(x)
        logits = self.output_head(x)
        return logits

# -----------------------------
# Utilities
# -----------------------------
def get_batch(data, batch_size, seq_len, device='cpu'):
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        # Fallback for small data
        actual_len = len(data) - 1
        x = data[:actual_len].unsqueeze(0).repeat(batch_size, 1).to(device)
        y = data[1:actual_len+1].unsqueeze(0).repeat(batch_size, 1).to(device)
        return x, y
    
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, eval_iters=50, batch_size=32, seq_len=64, device='cpu', vocab_size=50257):
    model.eval()
    losses = []
    for _ in range(min(eval_iters, len(data) // (batch_size * seq_len))):
        x, y = get_batch(data, batch_size, seq_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0

@torch.no_grad()
def sample(model, start_token, tokenizer, max_len=200, temperature=1.0, device='cpu'):
    model.eval()
    generated = [start_token]
    for _ in range(max_len):
        # RoPE handles relative position, so we just feed the context window
        context = torch.tensor([generated[-1024:]], dtype=torch.long, device=device)
        logits = model(context)
        logits = logits[0, -1] / temperature
        probs = F.softmax(logits, dim=0)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    model.train()
    return tokenizer.decode(generated)

def chat_mode(model, device, tokenizer):
    model.eval()
    print("\n" + "="*60 + "\nCHAT MODE (Type 'quit' to exit)\n" + "="*60)
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit']: break
            if not user_input: continue
            
            input_ids = tokenizer.encode(user_input)
            generated = input_ids.copy()
            
            print("Model: ", end="", flush=True)
            with torch.no_grad():
                for _ in range(200):
                    context = torch.tensor([generated[-1024:]], dtype=torch.long, device=device)
                    logits = model(context)
                    logits = logits[0, -1] / 0.8 # Slight temp reduction for chat
                    probs = F.softmax(logits, dim=0)
                    next_token = torch.multinomial(probs, 1).item()
                    generated.append(next_token)
                    
                    word = tokenizer.decode([next_token])
                    print(word, end="", flush=True)
                    if next_token == tokenizer.eos_token_id: break
            print("\n")
        except KeyboardInterrupt:
            break

# -----------------------------
# Training Logic with Elastic Context Curriculum
# -----------------------------
def train_on_parquet_files(model, optimizer, scheduler, data_loader, device, vocab_size, 
                           base_batch_size, base_seq_len, eval_interval, best_loss, tokenizer, scaler=None):
    
    total_files = data_loader.get_file_count()
    global_step = 0
    
    # Elastic Context Settings
    short_seq_len = base_seq_len
    long_seq_len = base_seq_len * 4  # Extended context (e.g., 2048)
    
    model.train()
    
    for epoch, (file_path, chunk_generator) in enumerate(data_loader.iterate_files(), 1):
        print(f"\n{'='*40}")
        print(f"File {epoch}/{total_files}: {Path(file_path).name}")
        print(f"{'='*40}")
        
        for raw_chunk in chunk_generator:
            # Split Data (90/10)
            split_idx = int(len(raw_chunk) * 0.9)
            train_chunk = raw_chunk[:split_idx]
            val_chunk = raw_chunk[split_idx:]
            
            if len(train_chunk) < short_seq_len + 1: continue
            
            # Calculate batches based on short len
            max_batches = (len(train_chunk) - short_seq_len - 1) // (base_batch_size * short_seq_len)
            
            for i in range(max_batches):
                optimizer.zero_grad(set_to_none=True)
                
                # --- ELASTIC CONTEXT CURRICULUM ---
                # 75% Short Context, 25% Long Context
                if random.random() < 0.25 and len(train_chunk) > long_seq_len + 1:
                    # Extended Context (Spectral Calibration)
                    curr_seq_len = long_seq_len
                    # Reduce batch size to prevent OOM
                    curr_batch_size = max(1, base_batch_size // 4)
                else:
                    # Standard Context
                    curr_seq_len = short_seq_len
                    curr_batch_size = base_batch_size
                # ----------------------------------
                
                x_batch, y_batch = get_batch(train_chunk, curr_batch_size, curr_seq_len, device)
                
                with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                    logits = model(x_batch)
                    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
                
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                if scheduler: scheduler.step()
                global_step += 1
                
                if global_step % eval_interval == 0:
                    # Validate (Standard Length)
                    val_loss = estimate_loss(model, val_chunk, eval_iters=20, batch_size=base_batch_size, 
                                           seq_len=short_seq_len, device=device, vocab_size=vocab_size)
                    
                    print(f"Step {global_step} | Train: {loss.item():.4f} | Val: {val_loss:.4f} | Ctx: {curr_seq_len}")
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        print(f"✨ New best loss! {best_loss:.4f}")
                        torch.save(model.state_dict(), 'best_model.pt')
                        print(sample(model, val_chunk[0].item(), tokenizer, max_len=100, device=device))
                        
    return best_loss

if __name__ == "__main__":
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # --- HYPERPARAMETERS ---
    d_model = 384
    n_frequencies = 256 
    n_layers = 6        
    n_heads = 6
    batch_size = 10     
    seq_len = 512       # Base sequence length
    learning_rate = 6e-4 
    
    model = HybridSpectralLM(vocab_size, d_model, n_frequencies, n_layers, n_heads).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, fused=(device=='cuda'))
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    
    if os.path.exists('best_model.pt'):
        print("Checkpoint found. (1) Load & Train, (2) Chat, (3) New?")
        ch = input("> ")
        if ch == '1': model.load_state_dict(torch.load('best_model.pt'))
        elif ch == '2': 
            model.load_state_dict(torch.load('best_model.pt'))
            chat_mode(model, device, tokenizer)
            exit()
            
    print("Select Data: (1) input.txt, (2) Parquet folder")
    mode = input("> ")
    
    if mode == '1':
        with open("input.txt", "r", encoding="utf-8") as f: text = f.read()
        full_data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        n = int(0.9 * len(full_data))
        train_data = full_data[:n]
        val_data = full_data[n:]
        
        print(f"Total tokens: {len(full_data):,}")
        print(f"Training on:  {len(train_data):,} tokens")
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000)
        
        best_loss = float('inf')
        
        # Elastic Curriculum for Single File Mode
        short_seq = seq_len
        long_seq = seq_len * 4
        
        for i in range(50000):
            # Elastic Context Logic
            if random.random() < 0.25 and len(train_data) > long_seq + 1:
                curr_seq = long_seq
                curr_bs = max(1, batch_size // 4)
            else:
                curr_seq = short_seq
                curr_bs = batch_size
                
            x, y = get_batch(train_data, curr_bs, curr_seq, device)
            
            with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if i % 500 == 0:
                val_loss = estimate_loss(model, val_data, batch_size=batch_size, seq_len=short_seq, device=device)
                print(f"Iter {i} | Train: {loss.item():.4f} | Val: {val_loss:.4f} | Ctx: {curr_seq}")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), 'best_model.pt')
                    print(sample(model, val_data[0].item(), tokenizer, max_len=100, device=device))

    elif mode == '2':
        folder = input("Folder path: ")
        col = input("Text column: ")
        loader = ParquetDataLoader(folder, col, tokenizer)
        train_on_parquet_files(model, optimizer, None, loader, device, vocab_size, 
                               batch_size, seq_len, 1000, float('inf'), tokenizer, scaler)