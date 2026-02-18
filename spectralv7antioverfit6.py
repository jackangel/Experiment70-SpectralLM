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
        self.threshold = threshold
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_(0, 0.02)
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        
    def forward(self, inputs):
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        if self.training:
            with torch.no_grad():
                encodings_sum = encodings.sum(0)
                
                self.ema_cluster_size.mul_(self.decay).add_(
                    encodings_sum, alpha=1 - self.decay
                )
                
                n = self.ema_cluster_size.sum()
                self.ema_cluster_size.add_(self.epsilon).div_(
                    n + self.num_embeddings * self.epsilon
                ).mul_(n)
                
                dw = torch.matmul(encodings.t(), flat_input)
                
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                
                self.embedding.weight.data.copy_(
                    self.ema_w / self.ema_cluster_size.unsqueeze(1)
                )

                dead_indices = (self.ema_cluster_size < self.threshold).nonzero(as_tuple=True)[0]
                
                if dead_indices.numel() > 0:
                    n_dead = dead_indices.numel()
                    
                    if flat_input.shape[0] > 0:
                        rand_idx = torch.randint(0, flat_input.shape[0], (n_dead,), device=flat_input.device)
                        new_embeddings = flat_input[rand_idx].detach()
                        
                        self.embedding.weight.data[dead_indices] = new_embeddings
                        
                        self.ema_cluster_size[dead_indices] = self.threshold
                        self.ema_w[dead_indices] = new_embeddings * self.threshold
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.view(inputs.shape[:-1])

# -----------------------------
# 2b. SIMPLIFIED: Adaptive Temporal VQ with Boundary Detection
# -----------------------------
class AdaptiveTemporalVQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, 
                 decay=0.99, epsilon=1e-5, threshold=1.0, min_span=2, max_span=8):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.min_span = min_span
        self.max_span = max_span
        
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, 
                                   decay, epsilon, threshold)
        
        # Simpler boundary predictor
        self.boundary_predictor = nn.Linear(embedding_dim, 1)
        
    def forward(self, x):
        B, T, D = x.shape
        device = x.device
        
        # Predict boundaries with simpler approach
        boundary_logits = self.boundary_predictor(x).squeeze(-1)  # [B, T]
        
        # Use straight-through estimator for training
        if self.training:
            # Soft boundaries for gradient flow
            boundary_probs = torch.sigmoid(boundary_logits)
            # Sample with temperature
            temperature = 0.5
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(boundary_probs) + 1e-10) + 1e-10)
            hard_boundaries = ((boundary_logits + gumbel_noise) / temperature > 0).float()
            # Straight-through: use hard for forward, soft for backward
            boundaries = hard_boundaries.detach() - boundary_probs.detach() + boundary_probs
        else:
            boundaries = (torch.sigmoid(boundary_logits) > 0.5).float()
        
        # Force boundaries at regular intervals (simplified segmentation)
        # This prevents the model from getting stuck
        fixed_segment_size = self.max_span
        num_segments = (T + fixed_segment_size - 1) // fixed_segment_size
        
        # Reshape and pool
        pad_length = num_segments * fixed_segment_size - T
        if pad_length > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_length))
        else:
            x_padded = x
        
        # Reshape to segments
        x_segments = x_padded.view(B, num_segments, fixed_segment_size, D)
        
        # Pool each segment (mean pooling for simplicity and speed)
        pooled = x_segments.mean(dim=2)  # [B, num_segments, D]
        
        # Quantize pooled segments
        quantized_pooled, vq_loss, indices = self.vq(pooled)
        
        # Expand back to original sequence length
        quantized_expanded = quantized_pooled.unsqueeze(2).expand(-1, -1, fixed_segment_size, -1)
        quantized_expanded = quantized_expanded.reshape(B, num_segments * fixed_segment_size, D)
        
        # Remove padding
        quantized_out = quantized_expanded[:, :T, :]
        
        # Expand indices similarly
        indices_expanded = indices.unsqueeze(2).expand(-1, -1, fixed_segment_size)
        indices_expanded = indices_expanded.reshape(B, num_segments * fixed_segment_size)
        indices_out = indices_expanded[:, :T]
        
        # Boundary loss (encourage sparse boundaries)
        target_boundary_rate = 1.0 / fixed_segment_size
        boundary_rate = boundaries.mean()
        boundary_loss = F.mse_loss(boundary_rate, torch.tensor(target_boundary_rate, device=device))
        
        total_loss = vq_loss + 0.01 * boundary_loss  # Reduced boundary loss weight
        
        return quantized_out, total_loss, indices_out, boundaries

# -----------------------------
# 2c. SIMPLIFIED: Relational Structure Learner
# -----------------------------
class RelationalStructureLearner(nn.Module):
    def __init__(self, d_model, num_relation_types=8):
        super().__init__()
        self.num_relation_types = num_relation_types
        
        # Simplified relation modeling
        self.relation_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x, window_size=3):
        B, T, D = x.shape
        device = x.device
        
        # Simple local context aggregation
        contexts = []
        for offset in range(1, min(window_size + 1, T)):
            # Shift and concatenate
            shifted = F.pad(x[:, offset:, :], (0, 0, 0, offset), value=0)
            combined = torch.cat([x, shifted], dim=-1)
            context = self.relation_encoder(combined)
            contexts.append(context / offset)  # Weight by distance
        
        if contexts:
            relation_context = torch.stack(contexts, dim=0).mean(dim=0)
        else:
            relation_context = torch.zeros_like(x)
        
        # Dummy relation logits for compatibility
        relation_logits = torch.zeros(B, max(0, T-1), self.num_relation_types, device=device)
        
        return relation_context, relation_logits

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
        
        A = decay * rot
        
        # Simplified convolution using FFT
        ones = torch.ones(1, 1, self.n_frequencies, device=device, dtype=A.dtype)
        A_expanded = A.view(1, 1, self.n_frequencies).expand(1, seq_len, self.n_frequencies)
        powers = torch.cumprod(torch.cat([ones, A_expanded[:, :-1, :]], dim=1), dim=1)
        kernel = powers * rot.view(1, 1, self.n_frequencies)
        
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
# 4. Spectral Bucket Model (Simplified)
# -----------------------------
class SpectralBucketLM(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_frequencies=128, 
                 num_buckets=512, vq_depth=2, num_relation_types=8, 
                 use_adaptive_vq=True, dropout=0.1):
        super().__init__()
        self.vq_depth = vq_depth
        self.num_buckets = num_buckets
        self.use_adaptive_vq = use_adaptive_vq
        self.num_relation_types = num_relation_types
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.bucket_dim = d_model // 2
        
        self.pre_vq_norm = nn.LayerNorm(d_model)
        self.pre_vq_proj = nn.Linear(d_model, self.bucket_dim)
        
        if use_adaptive_vq:
            self.vq_layers = nn.ModuleList([
                AdaptiveTemporalVQ(num_buckets, self.bucket_dim, min_span=2, max_span=8) 
                for _ in range(vq_depth)
            ])
        else:
            self.vq_layers = nn.ModuleList([
                VectorQuantizer(num_buckets, self.bucket_dim) for _ in range(vq_depth)
            ])
        
        self.post_vq_proj = nn.Linear(self.bucket_dim, d_model)

        self.relation_learner = RelationalStructureLearner(d_model, num_relation_types)
        
        self.bucket_embs = nn.ModuleList([
            nn.Embedding(num_buckets, d_model) for _ in range(vq_depth)
        ])
        
        self.bucket_projector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.bucket_layers = nn.ModuleList([
            SpectralBlock(d_model, n_frequencies, dropout) for _ in range(n_layers // 2)
        ])
        
        self.bucket_heads = nn.ModuleList([
            nn.Linear(d_model, num_buckets) for _ in range(vq_depth)
        ])
        
        self.relation_head = nn.Linear(d_model, num_relation_types)

        self.token_layers = nn.ModuleList([
            SpectralBlock(d_model, n_frequencies, dropout) for _ in range(n_layers)
        ])
        
        self.cross_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x):
        B, T = x.shape
        
        h_tokens = self.token_emb(x)
        h_tokens = self.dropout(h_tokens)
        
        h_norm = self.pre_vq_norm(h_tokens)
        h_projected = self.pre_vq_proj(h_norm)
        
        total_vq_loss = 0
        all_bucket_indices = []
        all_boundaries = []
        
        current_residual = h_projected
        
        for i in range(self.vq_depth):
            if self.use_adaptive_vq:
                quantized, loss, indices, boundaries = self.vq_layers[i](current_residual)
                all_boundaries.append(boundaries)
            else:
                quantized, loss, indices = self.vq_layers[i](current_residual)
            
            total_vq_loss = total_vq_loss + loss
            all_bucket_indices.append(indices)
            
            current_residual = current_residual - quantized
        
        bucket_indices = torch.stack(all_bucket_indices, dim=2)
        
        h_buckets = torch.zeros_like(h_tokens)
        
        for i in range(self.vq_depth):
            h_buckets = h_buckets + self.bucket_embs[i](all_bucket_indices[i])
        
        relation_context, relation_logits = self.relation_learner(h_buckets)
        
        bucket_states = []
        for layer in self.bucket_layers:
            h_buckets = layer(h_buckets)
            bucket_states.append(h_buckets)
            
        while len(bucket_states) < len(self.token_layers):
            bucket_states.append(h_buckets)
            
        bucket_logits_list = []
        for i in range(self.vq_depth):
            bucket_logits_list.append(self.bucket_heads[i](h_buckets))
        
        bucket_logits = torch.stack(bucket_logits_list, dim=2)
        
        relation_pred_logits = self.relation_head(h_buckets)
        
        aggregate_bucket_emb = torch.zeros_like(h_tokens)
        for i in range(self.vq_depth):
            aggregate_bucket_emb = aggregate_bucket_emb + self.bucket_embs[i](all_bucket_indices[i])
        
        combined_context = torch.cat([aggregate_bucket_emb, relation_context], dim=-1)
        initial_bucket_context = self.bucket_projector(combined_context)
        h_tokens = h_tokens + initial_bucket_context
        
        for i, layer in enumerate(self.token_layers):
            plan = bucket_states[i]
            injection = self.cross_projections[i](plan)
            h_tokens = layer(h_tokens, conditioning=injection)

        h_tokens = self.ln_f(h_tokens)
        token_logits = self.lm_head(h_tokens)
        
        return {
            'token_logits': token_logits,
            'bucket_logits': bucket_logits,
            'relation_logits': relation_logits,
            'relation_pred_logits': relation_pred_logits,
            'vq_loss': total_vq_loss,
            'bucket_indices': bucket_indices,
            'boundaries': all_boundaries if self.use_adaptive_vq else None
        }

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
        
        outputs = model(x)
        token_logits = outputs['token_logits']
        
        loss_token = F.cross_entropy(token_logits.view(-1, token_logits.size(-1)), y.view(-1))
        losses.append(loss_token.item())
    
    model.train()
    return sum(losses) / len(losses) if losses else 0.0

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="The", max_new_tokens=200, top_k=50):
    model.eval()
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = context
    for _ in range(max_new_tokens):
        outputs = model(generated)
        token_logits = outputs['token_logits']
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
    
    print(f"Starting Spectral Training (VQ Depth: {model.vq_depth}, Adaptive VQ: {model.use_adaptive_vq})...")
    
    step = 0
    while step < steps:
        if is_tensor:
            x, y = get_batch(train_data, batch_size, seq_len, device)
            
            if step == 0:
                print("Running first forward pass...")
            
            outputs = model(x)
            
            if step == 0:
                print("First forward pass completed!")
            
            token_logits = outputs['token_logits']
            bucket_logits = outputs['bucket_logits']
            vq_loss = outputs['vq_loss']
            bucket_indices = outputs['bucket_indices']
            relation_logits = outputs['relation_logits']
            relation_pred_logits = outputs['relation_pred_logits']
            
            loss_token = F.cross_entropy(token_logits.view(-1, token_logits.size(-1)), y.view(-1))
            
            bucket_targets = bucket_indices[:, 1:, :] 
            bucket_preds = bucket_logits[:, :-1, :, :]
            
            loss_bucket = torch.tensor(0.0, device=device)
            if bucket_targets.shape[1] > 0:
                for d in range(model.vq_depth):
                    pred_flat = bucket_preds[:, :, d, :].reshape(-1, model.num_buckets)
                    target_flat = bucket_targets[:, :, d].reshape(-1)
                    loss_bucket = loss_bucket + F.cross_entropy(pred_flat, target_flat)

            loss_relation = torch.tensor(0.0, device=device)
            if relation_logits.shape[1] > 0:
                relation_targets = torch.argmax(relation_logits[:, :, :], dim=-1)
                relation_preds = relation_pred_logits[:, :-1, :]
                
                if relation_targets.shape[1] > 0 and relation_preds.shape[1] > 0:
                    min_len = min(relation_targets.shape[1], relation_preds.shape[1])
                    loss_relation = F.cross_entropy(
                        relation_preds[:, :min_len, :].reshape(-1, model.num_relation_types),
                        relation_targets[:, :min_len].reshape(-1)
                    )
            
            total_entropy = 0
            for d in range(model.vq_depth):
                indices_d = bucket_indices[:, :, d]
                one_hot = F.one_hot(indices_d, num_classes=model.num_buckets).float()
                avg_usage = one_hot.mean(dim=(0, 1))
                entropy = -torch.sum(avg_usage * torch.log(avg_usage + 1e-10))
                total_entropy += entropy
            
            avg_entropy = total_entropy / model.vq_depth
            loss_entropy = -0.1 * avg_entropy

            loss = loss_token + (0.5 * loss_bucket) + (1.0 * vq_loss) + (0.05 * loss_relation) + loss_entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step += 1
            if step % 100 == 0:
                v_loss = estimate_loss(model, val_data, batch_size=batch_size, seq_len=seq_len, device=device) if is_tensor else 0.0
                
                uniq_l0 = torch.unique(bucket_indices[:,:,0]).numel()
                
                print(f"Step {step} | Loss: {loss.item():.4f} (Tok: {loss_token:.3f}, Bkt: {loss_bucket:.3f}, Rel: {loss_relation:.3f}) | Val: {v_loss:.4f}")
                print(f"   -> Active Buckets (L0): {uniq_l0}/{model.num_buckets} (Avg Entropy: {avg_entropy.item():.3f})")
                
                if model.use_adaptive_vq and outputs['boundaries'] is not None:
                    boundary_rate = outputs['boundaries'][0].mean().item()
                    print(f"   -> Boundary Rate: {boundary_rate:.3f}")

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
    print(f"Using device: {device}")
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    model = SpectralBucketLM(
        vocab_size=tokenizer.vocab_size, 
        d_model=384, 
        n_layers=6, 
        n_frequencies=256,  # Reduced from 256
        num_buckets=256,
        vq_depth=2,
        num_relation_types=8,
        use_adaptive_vq=True,
        dropout=0.2
    ).to(device)

    print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")

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
                for _ in range(200):
                    with torch.no_grad():
                        outputs = model(generated)
                        token_logits = outputs['token_logits']
                        logits = token_logits[:, -1, :] / temperature
                        
                        if top_k > 0:
                            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = -float('Inf')
                        
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
        if not os.path.exists("input.txt"):
            with open("input.txt", "w") as f:
                f.write("The quick brown fox jumps over the lazy dog. " * 1000)
                
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        train(model, data, tokenizer, steps=100000, batch_size=6, seq_len=512, device=device)
        
        print("\nFinal Generation Test:")
        generate_sample(model, tokenizer, device, prompt="The quick")