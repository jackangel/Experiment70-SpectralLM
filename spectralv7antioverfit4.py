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
# 2. Vector Quantizer (With EMA and Dead Bucket Restart)
# -----------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5, threshold=1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.threshold = threshold # Threshold for restarting dead buckets
        
        # Embedding weights (not trained via gradients when using EMA)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_(0, 0.02)
        
        # EMA cluster size tracking
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        # EMA embedding average
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        
    def forward(self, inputs):
        # inputs: [Batch, Seq, Dim]
        # Flatten input
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate distances: (x-y)^2 = x^2 + y^2 - 2xy
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Get encoding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Quantize
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # --- EMA UPDATE ---
        if self.training:
            with torch.no_grad():
                # Update cluster sizes with EMA
                # encodings: [N, num_embeddings] where N = batch*seq
                encodings_sum = encodings.sum(0)  # [num_embeddings]
                
                # Update the EMA of cluster size
                self.ema_cluster_size.mul_(self.decay).add_(
                    encodings_sum, alpha=1 - self.decay
                )
                
                # Laplace smoothing of the cluster size
                n = self.ema_cluster_size.sum()
                self.ema_cluster_size.add_(self.epsilon).div_(
                    n + self.num_embeddings * self.epsilon
                ).mul_(n)
                
                # Update embeddings with EMA
                # dw = sum of all inputs assigned to each embedding
                dw = torch.matmul(encodings.t(), flat_input)  # [num_embeddings, embedding_dim]
                
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                
                # Normalize embeddings by cluster size
                self.embedding.weight.data.copy_(
                    self.ema_w / self.ema_cluster_size.unsqueeze(1)
                )

                # --- RESTART DEAD BUCKETS ---
                # Check for buckets with usage below threshold
                dead_indices = (self.ema_cluster_size < self.threshold).nonzero(as_tuple=True)[0]
                
                if dead_indices.numel() > 0:
                    # Pick random inputs from the current batch to replace dead embeddings
                    n_dead = dead_indices.numel()
                    
                    # Ensure we have enough inputs to sample from
                    if flat_input.shape[0] > 0:
                        rand_idx = torch.randint(0, flat_input.shape[0], (n_dead,), device=flat_input.device)
                        new_embeddings = flat_input[rand_idx].detach()
                        
                        # Update weights
                        self.embedding.weight.data[dead_indices] = new_embeddings
                        
                        # Reset EMA statistics for these buckets so they don't die immediately
                        self.ema_cluster_size[dead_indices] = self.threshold
                        self.ema_w[dead_indices] = new_embeddings * self.threshold
        
        # Loss - only commitment loss since embedding is updated via EMA
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.view(inputs.shape[:-1])

# -----------------------------
# 3. Spectral Components
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
# 4. Spectral Bucket Model (Modified for Hierarchical VQ)
# -----------------------------
class SpectralBucketLM(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_frequencies=128, num_buckets=512, vq_depth=2, dropout=0.1):
        super().__init__()
        self.vq_depth = vq_depth
        self.num_buckets = num_buckets
        
        # A. Token Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # B. The Concept Quantizer (Buckets) - Hierarchical
        self.bucket_dim = d_model // 2
        
        self.pre_vq_norm = nn.LayerNorm(d_model)
        self.pre_vq_proj = nn.Linear(d_model, self.bucket_dim)
        
        # Create a list of VQ layers, one for each depth level
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(num_buckets, self.bucket_dim) for _ in range(vq_depth)
        ])
        
        self.post_vq_proj = nn.Linear(self.bucket_dim, d_model)

        # C. The "Bucket Stream"
        # We need separate embeddings for each level of the hierarchy to reconstruct the stream
        self.bucket_embs = nn.ModuleList([
            nn.Embedding(num_buckets, d_model) for _ in range(vq_depth)
        ])
        
        # Concept Projector
        self.bucket_projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

        self.bucket_layers = nn.ModuleList([
            SpectralBlock(d_model, n_frequencies, dropout) for _ in range(n_layers // 2)
        ])
        
        # Output heads for buckets - one for each depth level
        self.bucket_heads = nn.ModuleList([
            nn.Linear(d_model, num_buckets) for _ in range(vq_depth)
        ])

        # D. The "Token Stream"
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

        # E. Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x):
        B, T = x.shape
        
        # 1. Token Embeddings
        h_tokens = self.token_emb(x)
        h_tokens = self.dropout(h_tokens)
        
        # 2. Extract Buckets (Hierarchical / Residual VQ)
        h_norm = self.pre_vq_norm(h_tokens)
        h_projected = self.pre_vq_proj(h_norm)
        
        total_vq_loss = 0
        all_bucket_indices = []
        
        # Residual VQ Loop
        current_residual = h_projected
        
        for i in range(self.vq_depth):
            # Quantize current residual
            quantized, loss, indices = self.vq_layers[i](current_residual)
            
            total_vq_loss += loss
            all_bucket_indices.append(indices)
            
            # Update residual for next layer: residual = residual - quantized
            # Note: quantized has gradients via STE, so gradients flow back to h_projected
            current_residual = current_residual - quantized
        
        # Stack indices: [B, T, Depth]
        bucket_indices = torch.stack(all_bucket_indices, dim=2)
        
        # 3. Process Bucket Stream
        # Reconstruct aggregate embedding from all levels
        h_buckets = torch.zeros_like(h_tokens)
        
        # Sum embeddings from all levels corresponding to the indices found
        for i in range(self.vq_depth):
            h_buckets = h_buckets + self.bucket_embs[i](all_bucket_indices[i])
        
        bucket_states = []
        for layer in self.bucket_layers:
            h_buckets = layer(h_buckets)
            bucket_states.append(h_buckets)
            
        while len(bucket_states) < len(self.token_layers):
            bucket_states.append(h_buckets)
            
        # Predict next buckets for ALL levels
        bucket_logits_list = []
        for i in range(self.vq_depth):
            bucket_logits_list.append(self.bucket_heads[i](h_buckets))
        
        # Stack logits: [B, T, Depth, Num_Buckets]
        bucket_logits = torch.stack(bucket_logits_list, dim=2)
        
        # 4. Process Token Stream
        # Initial context is derived from the aggregate bucket info
        # We reconstruct the context from the embeddings we just found
        aggregate_bucket_emb = torch.zeros_like(h_tokens)
        for i in range(self.vq_depth):
            aggregate_bucket_emb = aggregate_bucket_emb + self.bucket_embs[i](all_bucket_indices[i])
            
        initial_bucket_context = self.bucket_projector(aggregate_bucket_emb)
        h_tokens = h_tokens + initial_bucket_context
        
        for i, layer in enumerate(self.token_layers):
            plan = bucket_states[i]
            injection = self.cross_projections[i](plan)
            h_tokens = layer(h_tokens, conditioning=injection)

        h_tokens = self.ln_f(h_tokens)
        token_logits = self.lm_head(h_tokens)
        
        return token_logits, bucket_logits, total_vq_loss, bucket_indices

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
        
        token_logits, bucket_logits, vq_loss, bucket_indices = model(x)
        
        loss_token = F.cross_entropy(token_logits.view(-1, token_logits.size(-1)), y.view(-1))
        
        # bucket_indices: [B, T, Depth]
        # bucket_logits: [B, T, Depth, Num_Buckets]
        
        bucket_targets = bucket_indices[:, 1:, :] # Shift right
        bucket_preds = bucket_logits[:, :-1, :, :] # Shift left
        
        loss_bucket = 0
        if bucket_targets.shape[1] > 0:
            # Sum loss across all depth levels
            for d in range(model.vq_depth):
                pred_flat = bucket_preds[:, :, d, :].reshape(-1, model.num_buckets)
                target_flat = bucket_targets[:, :, d].reshape(-1)
                loss_bucket += F.cross_entropy(pred_flat, target_flat)
        
        losses.append((loss_token + loss_bucket + vq_loss).item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="The", max_new_tokens=200, top_k=50):
    model.eval()
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = context
    for _ in range(max_new_tokens):
        token_logits, _, _, _ = model(generated)
        logits = token_logits[:, -1, :]
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
    
    print(f"Starting Spectral Training (VQ Depth: {model.vq_depth})...")
    
    step = 0
    while step < steps:
        if is_tensor:
            x, y = get_batch(train_data, batch_size, seq_len, device)
            batches = [(x, y)]
        else:
            pass 
            
        for x, y in batches:
            token_logits, bucket_logits, vq_loss, bucket_indices = model(x)
            
            # 1. Token Loss
            loss_token = F.cross_entropy(token_logits.view(-1, token_logits.size(-1)), y.view(-1))
            
            # 2. Bucket Sequence Loss (Summed over Depth)
            bucket_targets = bucket_indices[:, 1:, :] 
            bucket_preds = bucket_logits[:, :-1, :, :]
            
            loss_bucket = torch.tensor(0.0, device=device)
            if bucket_targets.shape[1] > 0:
                for d in range(model.vq_depth):
                    pred_flat = bucket_preds[:, :, d, :].reshape(-1, model.num_buckets)
                    target_flat = bucket_targets[:, :, d].reshape(-1)
                    loss_bucket = loss_bucket + F.cross_entropy(pred_flat, target_flat)

            # 3. Entropy Regularization (Prevent Collapse) - Averaged over Depth
            # bucket_indices: [B, T, Depth]
            total_entropy = 0
            for d in range(model.vq_depth):
                indices_d = bucket_indices[:, :, d]
                one_hot = F.one_hot(indices_d, num_classes=model.num_buckets).float()
                avg_usage = one_hot.mean(dim=(0, 1)) # [Num_Buckets]
                entropy = -torch.sum(avg_usage * torch.log(avg_usage + 1e-10))
                total_entropy += entropy
            
            # Average entropy across levels for logging/loss scaling
            avg_entropy = total_entropy / model.vq_depth
            loss_entropy = -0.5 * avg_entropy #tweaking point

            # Total Loss (tweaking point)
            loss = loss_token + (1.0 * loss_bucket) + (1.0 * vq_loss) + loss_entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step += 1
            if step % 100 == 0:
                v_loss = estimate_loss(model, val_data, batch_size=batch_size, seq_len=seq_len, device=device) if is_tensor else 0.0
                
                # Calculate active buckets for the first layer just for display
                uniq_l0 = torch.unique(bucket_indices[:,:,0]).numel()
                
                print(f"Step {step} | Loss: {loss.item():.4f} (Tok: {loss_token:.3f}, Bkt: {loss_bucket:.3f}) | Val: {v_loss:.4f}")
                print(f"   -> Active Buckets (L0): {uniq_l0}/{model.num_buckets} (Avg Entropy: {avg_entropy.item():.3f})")

            if step % 1000 == 0:
                print(f"\n--- Prediction at Step {step} ---")
                sample_text = generate_sample(model, tokenizer, device, prompt="The", top_k=10)
                print(f"Generated: {sample_text}\n-------------------------------")
            
            if step >= steps: break
    
    print("Training finished. Saving checkpoint...")
    torch.save(model.state_dict(), "checkpoint.pt")
    print("Checkpoint saved to checkpoint.pt")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Initialize Spectral Bucket Model with VQ Depth
    model = SpectralBucketLM(
        vocab_size=tokenizer.vocab_size, 
        d_model=384, 
        n_layers=6, 
        n_frequencies=256,
        num_buckets=256,
        vq_depth=2,  # Configurable Depth for VQ Hierarchy
        dropout=0.2
    ).to(device)

    if os.path.exists("checkpoint.pt"):
        print("Checkpoint found. Loading and entering Chat Mode...")
        model.load_state_dict(torch.load("checkpoint.pt", map_location=device))
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
                        token_logits, _, _, _ = model(generated)
                        logits = token_logits[:, -1, :] / temperature
                        
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
                        
                        # Stop if EOS token (optional, depending on tokenizer)
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                            
                        print(tokenizer.decode([next_token.item()]), end="", flush=True)
                print()
                
            except KeyboardInterrupt:
                print("\nExiting Chat Mode.")
                break
    else:
        # Dummy Data Generation
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
            token_logits, _, _, _ = model(context)
            next_token = torch.argmax(token_logits[0, -1]).item()
            context = torch.cat([context, torch.tensor([[next_token]], device=device)], dim=1)
            print(tokenizer.decode([next_token]), end="", flush=True)
        print()