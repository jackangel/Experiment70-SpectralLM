import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from transformers import GPT2TokenizerFast

# -----------------------------
# Load data and setup tokenizer
# -----------------------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# -----------------------------
# Cross-Attention Module
# -----------------------------
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Query from current layer, Key/Value from previous layer
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key_value):
        """
        query: [batch, seq_len, d_model] - from current layer
        key_value: [batch, seq_len, d_model] - from previous layer
        """
        batch_size, seq_len, _ = query.shape
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask (attend only to past)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out

# -----------------------------
# Spectral Layer with Cross-Attention
# -----------------------------
class SpectralLayerWithAttention(nn.Module):
    def __init__(self, d_model, n_frequencies, n_heads=4, use_cross_attn=False):
        super().__init__()
        
        self.d_model = d_model
        self.n_frequencies = n_frequencies
        self.use_cross_attn = use_cross_attn
        
        # Project to spectral space
        self.to_spectral = nn.Linear(d_model, n_frequencies * 2, bias=False)
        
        # Spectral parameters
        self.log_decay = nn.Parameter(torch.zeros(n_frequencies))
        self.frequencies = nn.Parameter(torch.randn(n_frequencies) * 0.01)
        
        # Project back
        self.from_spectral = nn.Linear(n_frequencies * 2, d_model, bias=False)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Cross-attention (if enabled)
        if use_cross_attn:
            self.cross_attn = CrossAttention(d_model, n_heads)
            self.norm_cross = nn.LayerNorm(d_model)
        
        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )
        
        self.register_buffer('dt', torch.tensor(1.0))
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.to_spectral.weight, gain=0.1)
        nn.init.xavier_uniform_(self.from_spectral.weight, gain=0.1)
        
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x, prev_layer_output=None):
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # 1. Spectral processing
        residual = x
        x = self.norm1(x)
        
        spectral = self.to_spectral(x)
        real = spectral[..., :self.n_frequencies]
        imag = spectral[..., self.n_frequencies:]
        
        # Rotation parameters
        decay = torch.sigmoid(self.log_decay)
        omega = torch.tanh(self.frequencies) * 0.1
        cos_omega = torch.cos(omega * self.dt)
        sin_omega = torch.sin(omega * self.dt)
        
        # Recurrent processing
        state_real = torch.zeros(batch_size, self.n_frequencies, device=device)
        state_imag = torch.zeros(batch_size, self.n_frequencies, device=device)
        
        outputs_real = []
        outputs_imag = []
        
        for t in range(seq_len):
            state_real = state_real * decay
            state_imag = state_imag * decay
            
            state_real = state_real + real[:, t] * 0.1
            state_imag = state_imag + imag[:, t] * 0.1
            
            new_real = state_real * cos_omega - state_imag * sin_omega
            new_imag = state_real * sin_omega + state_imag * cos_omega
            
            state_real = new_real
            state_imag = new_imag
            
            state_real = torch.clamp(state_real, -10, 10)
            state_imag = torch.clamp(state_imag, -10, 10)
            
            outputs_real.append(state_real)
            outputs_imag.append(state_imag)
        
        out_real = torch.stack(outputs_real, dim=1)
        out_imag = torch.stack(outputs_imag, dim=1)
        
        spectral_out = torch.cat([out_real, out_imag], dim=-1)
        x = self.from_spectral(spectral_out)
        
        x = residual + x * 0.1
        
        # 2. Cross-attention to previous layer (if available)
        if self.use_cross_attn and prev_layer_output is not None:
            residual2 = x
            x = self.norm_cross(x)
            x = self.cross_attn(x, prev_layer_output)
            x = residual2 + x * 0.1
        
        # 3. FFN
        residual3 = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual3 + x * 0.1
        
        return x

# -----------------------------
# Hybrid Spectral-Attention Model
# -----------------------------
class HybridSpectralLM(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_frequencies=64, n_layers=4, n_heads=4):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_frequencies = n_frequencies
        self.n_layers = n_layers
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        
        # Spectral layers with cross-attention
        self.layers = nn.ModuleList([
            SpectralLayerWithAttention(
                d_model, 
                n_frequencies, 
                n_heads,
                use_cross_attn=(i > 0)  # Skip cross-attn on first layer
            )
            for i in range(n_layers)
        ])
        
        # Output
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.output_head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        for layer in self.layers:
            layer._init_weights()
    
    def forward(self, sequence):
        batch_size, seq_len = sequence.shape
        
        # Embed tokens with position
        x = self.token_embedding(sequence)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Apply layers with cross-attention
        prev_output = None
        for layer in self.layers:
            x = layer(x, prev_output)
            prev_output = x  # Pass to next layer for cross-attention
        
        # Output
        x = self.output_norm(x)
        logits = self.output_head(x)
        
        return logits

# -----------------------------
# Training utilities
# -----------------------------
def get_batch(data, batch_size, seq_len, device='cpu'):
    max_start = len(data) - seq_len - 1
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def sample(model, start_token, max_len=200, temperature=1.0, device='cpu'):
    model.eval()
    generated = [start_token]
    
    for _ in range(max_len):
        context = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
        logits = model(context)
        logits = logits[0, -1] / temperature
        
        if torch.isnan(logits).any():
            break
        
        probs = F.softmax(logits, dim=0)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
    
    model.train()
    return tokenizer.decode(generated)

@torch.no_grad()
def estimate_loss(model, data, eval_iters=50):
    model.eval()
    losses = []
    
    for _ in range(eval_iters):
        x, y = get_batch(data, 32, 64, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses)

# -----------------------------
# Chat mode function
# -----------------------------
def chat_mode(model, device):
    """Interactive chat mode with the model"""
    model.eval()
    print("\n" + "="*60)
    print("CHAT MODE - Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting chat mode...")
                break
            
            if not user_input:
                continue
            
            # Encode the input
            input_ids = tokenizer.encode(user_input)
            generated = input_ids.copy()
            
            # Generate response
            with torch.no_grad():
                for i in range(500):  # Increased max tokens
                    context = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
                    logits = model(context)
                    logits = logits[0, -1] / 1.0  # temperature
                    
                    if torch.isnan(logits).any():
                        break
                    
                    probs = F.softmax(logits, dim=0)
                    next_token = torch.multinomial(probs, 1).item()
                    generated.append(next_token)
                    
                    # Stop only at EOS token, or after generating enough text
                    # Allow multiple sentences/paragraphs
                    if next_token == tokenizer.eos_token_id:
                        break
                    
                    # Stop after generating at least 50 tokens if we hit a period followed by space
                    # This allows for complete thoughts
                    if i > 50 and len(generated) >= 2:
                        decoded_so_far = tokenizer.decode(generated[-10:])
                        # Check for sentence endings: ". ", "! ", "? " followed by potential end
                        if any(end in decoded_so_far for end in ['. ', '.\n', '! ', '!\n', '? ', '?\n']):
                            # Look ahead - if the last few tokens suggest completion, stop
                            recent_text = tokenizer.decode(generated[-3:]).strip()
                            if recent_text and recent_text[-1] in '.!?':
                                break
            
            # Decode the full generated text (including input)
            full_text = tokenizer.decode(generated)
            
            # Extract only the new generated part (after the input)
            response = full_text[len(user_input):].strip()
            
            print(f"Model: {response}\n")
            
        except KeyboardInterrupt:
            print("\nExiting chat mode...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

# -----------------------------
# Main execution
# -----------------------------
print(f"Vocabulary size: {vocab_size}")
print(f"Dataset size: {len(data)} tokens")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Hyperparameters
d_model = 384
n_frequencies = 64
n_layers = 4
n_heads = 4
batch_size = 32
seq_len = 128
learning_rate = 3e-4
num_iters = 100000
eval_interval = 200

# Initialize model
model = HybridSpectralLM(vocab_size, d_model, n_frequencies, n_layers, n_heads).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Architecture: {n_layers} spectral layers + cross-attention")
print(f"Model size: d_model={d_model}, n_heads={n_heads}\n")

# -----------------------------
# Checkpoint handling
# -----------------------------
checkpoint_path = 'best_model.pt'
start_iteration = 0

if os.path.exists(checkpoint_path):
    print("="*60)
    print(f"Found checkpoint: {checkpoint_path}")
    print("="*60)
    print("\nOptions:")
    print("  1. Continue training from checkpoint")
    print("  2. Enter chat mode")
    print("  3. Start fresh training (ignore checkpoint)")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            print(f"\nLoading checkpoint from {checkpoint_path}...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Checkpoint loaded successfully! Continuing training...\n")
            break
        elif choice == '2':
            print(f"\nLoading checkpoint from {checkpoint_path}...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Checkpoint loaded successfully!\n")
            chat_mode(model, device)
            exit(0)
        elif choice == '3':
            print("\nStarting fresh training (checkpoint ignored)...\n")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# -----------------------------
# Training
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters)

if device == 'cuda':
    torch.backends.cudnn.benchmark = True

import time
start_time = time.time()
tokens_processed = 0
best_loss = float('inf')

for iteration in range(num_iters):
    x_batch, y_batch = get_batch(data, batch_size, seq_len, device)
    tokens_processed += batch_size * seq_len
    
    logits = model(x_batch)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
    
    if torch.isnan(loss) or loss.item() > 50:
        print(f"Unstable loss at iteration {iteration}: {loss.item():.4f}")
        print("Reinitializing model...")
        model._init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters - iteration)
        continue
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    if iteration % eval_interval == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_processed / elapsed
        
        val_loss = estimate_loss(model, data)
        
        print(f"\n{'='*60}")
        print(f"Iter {iteration}/{num_iters}")
        print(f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
        print(f"Speed: {tokens_per_sec:.0f} tokens/sec | Time: {elapsed:.1f}s")
        print(f"{'='*60}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"✨ New best loss! {best_loss:.4f}")
            torch.save(model.state_dict(), 'best_model.pt')
        
        start_idx = data[0].item()
        sample_text = sample(model, start_idx, max_len=300, temperature=0.8, device=device)
        print(sample_text[:500])
        print()

print("\n" + "="*60)
print("Training complete!")
print(f"Best validation loss: {best_loss:.4f}")
print("="*60)