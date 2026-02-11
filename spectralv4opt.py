import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import glob
import time
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
# Causal Attention (Hybrid: Self or Cross)
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
        
        q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
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
# Robust Spectral Layer
# -----------------------------
class SpectralLayerWithAttention(nn.Module):
    def __init__(self, d_model, n_frequencies, n_heads=4):
        super().__init__()
        
        self.d_model = d_model
        self.n_frequencies = n_frequencies
        
        # Spectral Projection
        self.to_spectral = nn.Linear(d_model, n_frequencies * 2, bias=False)
        
        # Learnable Decay and Frequencies
        # We parameterize decay to ensure it stays between 0 and 1
        self.log_decay = nn.Parameter(torch.linspace(-3.0, -0.5, n_frequencies)) 
        self.frequencies = nn.Parameter(torch.randn(n_frequencies))
        
        self.from_spectral = nn.Linear(n_frequencies * 2, d_model, bias=False)
        
        # Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Attention
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
        Replaces the Python loop with O(L log L) FFT Convolution.
        Mathematically equivalent to the recurrence:
        h_t = (decay * e^{i*omega}) * h_{t-1} + (e^{i*omega}) * x_t
        """
        batch_size, seq_len, n_freq = inputs_real.shape
        device = inputs_real.device
        
        # 1. Create the Complex Input (B, L, F)
        u = torch.complex(inputs_real, inputs_imag)
        
        # 2. Construct the Convolution Kernel (Filter)
        # The effective recurrence multiplier is (decay * e^{i*omega})
        # The input multiplier is (e^{i*omega}) due to the rotation happening after addition
        
        # Complex rotation term e^{i*omega}
        rot = torch.complex(torch.cos(omega), torch.sin(omega)) # Shape: (F)
        
        # Combined Step Multiplier: A = decay * rot
        A = decay * rot # Shape: (F)
        
        # We need powers of A: [A^0, A^1, A^2, ... A^{L-1}]
        # But efficiently: we construct the range [0, 1, ..., L-1] and broadcast
        t = torch.arange(seq_len, device=device).float().unsqueeze(-1) # (L, 1)
        
        # Kernel K[t] = A^t * rot
        # Note: In log space for numerical stability: log(A^t) = t * log(A)
        # However, since A is complex, we just use cumprod for simplicity/speed on L=512
        
        # Prepare A for cumprod broadcast (1, L, F)
        A_expanded = A.view(1, 1, n_freq).expand(1, seq_len, n_freq)
        
        # Compute powers using cumulative product: [A, A^2, A^3...]
        # We prepend 1 to get [1, A, A^2...]
        ones = torch.ones(1, 1, n_freq, device=device, dtype=A.dtype)
        powers = torch.cumprod(torch.cat([ones, A_expanded[:, :-1, :]], dim=1), dim=1)
        
        # The Impulse Response Kernel
        # In the loop: state = (state + input) * rot * decay (approx)
        # Let's match the exact loop logic:
        # s_t = (s_{t-1} * decay + x_t) * rot
        # s_t = s_{t-1} * (decay * rot) + x_t * rot
        # This means the kernel is powers of (decay*rot) multiplied by (rot)
        kernel = powers * rot.view(1, 1, n_freq) # (1, L, F)
        
        # 3. Perform Convolution via FFT
        # We pad to L + L = 2L to perform "Linear Convolution" using FFT (which does Circular)
        n_fft = 2 * seq_len
        
        # FFT over the sequence dimension (dim=1)
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
        # 1. Spectral Block
        residual = x
        x_norm = self.norm1(x)
        
        spectral = self.to_spectral(x_norm)
        real = spectral[..., :self.n_frequencies]
        imag = spectral[..., self.n_frequencies:]
        
        # Parameters
        decay = torch.sigmoid(self.log_decay)
        omega = self.frequencies * 0.1 * self.dt
        
        # --- FAST FFT PATH ---
        out_real, out_imag = self.fft_convolution(real, imag, decay, omega)
        # ---------------------
        
        spectral_out = torch.cat([out_real, out_imag], dim=-1)
        x_spectral = self.from_spectral(spectral_out)
        
        x = residual + x_spectral
        
        # 2. Attention Block
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
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            SpectralLayerWithAttention(d_model, n_frequencies, n_heads)
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
        x = x + self.pos_embedding[:, :seq_len, :]
        
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
        x = data[:seq_len].unsqueeze(0).repeat(batch_size, 1).to(device)
        y = data[1:seq_len+1].unsqueeze(0).repeat(batch_size, 1).to(device)
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
        context = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
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
                    context = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
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
# Training Logic
# -----------------------------
def train_on_parquet_files(model, optimizer, scheduler, data_loader, device, vocab_size, 
                           batch_size, seq_len, eval_interval, best_loss, tokenizer, scaler=None):
    
    total_files = data_loader.get_file_count()
    
    for epoch, (file_path, chunk_generator) in enumerate(data_loader.iterate_files(), 1):
        print(f"\nProcessing File {epoch}/{total_files}: {Path(file_path).name}")
        
        for chunk_data in chunk_generator:
            if len(chunk_data) < seq_len + 1: continue
            
            max_batches = (len(chunk_data) - seq_len - 1) // (batch_size * seq_len)
            if max_batches == 0: continue
            
            for i in range(max_batches):
                x_batch, y_batch = get_batch(chunk_data, batch_size, seq_len, device)
                
                # Mixed Precision
                with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                    logits = model(x_batch)
                    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
                
                optimizer.zero_grad(set_to_none=True)
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
                
                if i % eval_interval == 0:
                    val_loss = estimate_loss(model, chunk_data, eval_iters=20, batch_size=batch_size, 
                                           seq_len=seq_len, device=device, vocab_size=vocab_size)
                    print(f"Iter {i} | Train: {loss.item():.4f} | Val: {val_loss:.4f}")
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        print(f"✨ New best loss! {best_loss:.4f}")
                        torch.save(model.state_dict(), 'best_model.pt')
                        # Sample to verify quality
                        print(sample(model, chunk_data[0].item(), tokenizer, max_len=100, device=device))
    return best_loss

if __name__ == "__main__":
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # --- UPDATED HYPERPARAMETERS FOR PATH B ---
    d_model = 384
    n_frequencies = 256 # Slightly reduced to allow larger batch/seq
    n_layers = 6        # Deeper model
    n_heads = 6
    batch_size = 10     # Reduced batch size to fit 512 seq_len in VRAM
    seq_len = 512       # CRITICAL: Increased context window
    learning_rate = 6e-4 # Slightly higher start for spectral convergence
    
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
        
        # --- DATA SPLIT FIX ---
        n = int(0.9 * len(full_data)) # 90% Train, 10% Validation
        train_data = full_data[:n]
        val_data = full_data[n:]
        
        print(f"Total tokens: {len(full_data):,}")
        print(f"Training on:  {len(train_data):,} tokens")
        print(f"Validating on: {len(val_data):,} tokens")
        # ----------------------
        
        # Scheduler for standard training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000)
        
        best_loss = float('inf')
        for i in range(50000):
            # Train on train_data
            x, y = get_batch(train_data, batch_size, seq_len, device)
            
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
                # Validate on val_data
                val_loss = estimate_loss(model, val_data, batch_size=batch_size, seq_len=seq_len, device=device)
                print(f"Iter {i} | Train: {loss.item():.4f} | Val: {val_loss:.4f}")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), 'best_model.pt')
                    # Sample from validation data start to see if it generalizes
                    print(sample(model, val_data[0].item(), tokenizer, max_len=100, device=device))

    elif mode == '2':
        folder = input("Folder path: ")
        col = input("Text column: ")
        loader = ParquetDataLoader(folder, col, tokenizer)
        # Note: Parquet mode splits are usually handled by file separation or custom logic
        train_on_parquet_files(model, optimizer, None, loader, device, vocab_size, 
                               batch_size, seq_len, 1000, float('inf'), tokenizer, scaler)