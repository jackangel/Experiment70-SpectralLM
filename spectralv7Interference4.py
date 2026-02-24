import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import tiktoken
import numpy as np
import random

# ==========================================
# ADAPTIVE LOSS WEIGHTING MODULE
# ==========================================
class AdaptiveWeightedLoss(nn.Module):
    """
    Learns optimal loss weights using uncertainty weighting (Kendall et al.)
    Each loss gets a learnable log-variance parameter that balances its contribution.
    """
    def __init__(self, num_losses=3):  # auto, spectral, memory
        super().__init__()
        # Log-variance parameters (ensures positive weights via exp)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses):
        """
        Args:
            losses: dict with keys 'auto', 'spectral', 'memory', 'bert' containing loss tensors
        Returns:
            Combined weighted loss
        """
        weighted_losses = []
        loss_names = ['auto', 'spectral', 'memory']
        
        for i, name in enumerate(loss_names):
            if name in losses and losses[name] is not None:
                # Precision = exp(-log_var), inverse of variance
                precision = torch.exp(-self.log_vars[i])
                # Weighted loss + regularization term
                weighted_loss = precision * losses[name] + self.log_vars[i]
                weighted_losses.append(weighted_loss)
        
        return sum(weighted_losses) if weighted_losses else torch.tensor(0.0)
    
    def get_weights(self):
        """Return current effective weights for logging"""
        with torch.no_grad():
            weights = torch.exp(-self.log_vars)
            return {
                'w_auto': weights[0].item(),
                'w_spectral': weights[1].item(),
                'w_memory': weights[2].item(),
            }

# ==========================================
# 1. Hyperparameters & Configuration
# ==========================================
# I/O
out_dir = 'out'
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
eval_interval = 5000
log_interval = 100
eval_iters = 50
always_save_checkpoint = True 

# Data
batch_size = 2       
block_size = 1024     
dataset = 'input.txt' 

# Model Architecture
n_layer = 8          # More layers for spectral
n_head = 8
n_embd = 512         
dropout = 0.2
label_smoothing = 0.1

# Enhanced Spectral Settings
n_frequencies = 768       # Increased from 512
spectral_decay_init = -3.0  
spectral_freq_scale = 0.1   
tokens_to_predict = 5

# Memory Tokens Settings
n_memory_tokens = 16

# Sliding Window Settings
window_stride = 8

# BERT-style Training Settings
bert_every_n_steps = 5
bert_mask_prob = 0.15

# Repetition Penalty Settings (ONLY for generation)
repetition_penalty_generate = 1.5
repetition_context_size = 30

# Optimizer
learning_rate = 3e-4
max_iters = 200000
lr_decay_iters = 5000
min_lr = 3e-5
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95 
grad_clip = 1.0

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
random.seed(1337)

# ==========================================
# 2. Data Loader
# ==========================================
print(f"Loading text from {dataset}...")
try:
    with open(dataset, 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print(f"ERROR: {dataset} not found. Creating dummy text.")
    text = "The quick brown fox jumps over the lazy dog. " * 10000

enc = tiktoken.get_encoding("gpt2")
vocab_size = 50304 
train_ids = enc.encode(text)
print(f"Total Tokens: {len(train_ids)}")

data = torch.tensor(train_ids, dtype=torch.long)
n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split, mask_for_bert=False):
    data_src = train_data if split == 'train' else val_data
    if len(data_src) <= block_size:
        raise ValueError("Data is too short.")
    ix = torch.randint(len(data_src) - block_size, (batch_size,))
    x = torch.stack([data_src[i:i+block_size] for i in ix])
    y = torch.stack([data_src[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    
    if not mask_for_bert:
        return x, y
    
    x_masked = x.clone()
    bert_labels = torch.full_like(x, -100)
    
    mask = torch.rand(x.shape, device=device) < bert_mask_prob
    masked_positions = mask.nonzero(as_tuple=False)
    
    for batch_idx, pos_idx in masked_positions:
        batch_idx = batch_idx.item()
        pos_idx = pos_idx.item()
        bert_labels[batch_idx, pos_idx] = x[batch_idx, pos_idx]
        
        rand = torch.rand(1).item()
        if rand < 0.8:
            x_masked[batch_idx, pos_idx] = vocab_size
        elif rand < 0.9:
            x_masked[batch_idx, pos_idx] = torch.randint(0, vocab_size, (1,)).item()
    
    return x_masked, bert_labels, y

# ==========================================
# 3. Model Components
# ==========================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=65536, theta=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._update_cos_sin_tables(max_seq_len)

    def _update_cos_sin_tables(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.cos_cached.shape[2]:
            self._update_cos_sin_tables(seq_len)
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# ==========================================
# ENHANCED Spectral State Space Block
# ==========================================
class EnhancedSpectralBlock(nn.Module):
    """Enhanced with more frequencies and better stability"""
    def __init__(self, d_model, n_frequencies, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.n_frequencies = n_frequencies
        self.max_seq_len = max_seq_len
        
        # Project to spectral domain
        self.to_spectral = nn.Linear(d_model, n_frequencies * 2, bias=False)
        
        # Frequency forget gates
        self.freq_forget_gate = nn.Sequential(
            nn.Linear(d_model, n_frequencies),
            nn.Sigmoid()
        )
        
        # Adaptive frequency scaling
        self.freq_importance = nn.Parameter(torch.ones(n_frequencies))
        
        # Frequency grouping (more bands for higher resolution)
        self.n_bands = 6  # Increased from 4
        self.band_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            ) for _ in range(self.n_bands)
        ])
        
        # State space parameters
        freq_groups = torch.linspace(spectral_decay_init, -1.0, n_frequencies)
        self.log_decay = nn.Parameter(freq_groups)
        
        freq_init = torch.zeros(n_frequencies)
        for i in range(n_frequencies):
            freq_init[i] = (i / n_frequencies) * 2.0 - 1.0
        self.frequencies = nn.Parameter(freq_init * 0.1)
        
        # Temporal context gate
        self.temporal_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.dt = nn.Parameter(torch.ones(n_frequencies) * 0.3)
        
        # Concept preservation
        self.concept_proj = nn.Linear(d_model, d_model)
        self.concept_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.gate = nn.Linear(d_model, d_model)
        self.from_spectral = nn.Linear(n_frequencies * 2, d_model, bias=False)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.3)
        
        # Gate temperature
        self.gate_temperature = nn.Parameter(torch.ones(1) * 0.5)
        
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), 
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        device = x.device
        original_dtype = x.dtype
        
        residual = x
        x_norm = self.norm(x)
        
        # Concept preservation
        concept = self.concept_proj(x_norm)
        concept_strength = self.concept_gate(x_norm)
        concept_preserved = concept * concept_strength
        
        # Forget gates with temperature
        forget_gates = self.freq_forget_gate(x_norm)
        forget_gates = torch.sigmoid(torch.log(forget_gates + 1e-8) / self.gate_temperature)
        
        temporal_strength = self.temporal_gate(x_norm)
        temporal_strength = torch.sigmoid(torch.log(temporal_strength + 1e-8) / self.gate_temperature)
        
        # Band-specific gates
        band_size = self.n_frequencies // self.n_bands
        band_strengths = []
        for i, gate in enumerate(self.band_gates):
            band_strengths.append(gate(x_norm))
        band_strengths = torch.cat(band_strengths, dim=-1)
        
        # Project to spectral domain
        spectral = self.to_spectral(x_norm)
        real = spectral[..., :self.n_frequencies]
        imag = spectral[..., self.n_frequencies:]
        
        real_f32 = real.float()
        imag_f32 = imag.float()
        u = torch.complex(real_f32, imag_f32)
        
        # Apply forget gates
        forget_gates_complex = forget_gates.float()
        u = u * forget_gates_complex
        
        # Construct state space kernel
        decay_base = torch.sigmoid(self.log_decay.float())
        decay_base = torch.clamp(decay_base, min=0.1, max=0.9)
        
        freq_importance_normalized = F.softmax(self.freq_importance.float(), dim=0) * self.n_frequencies
        decay = decay_base * freq_importance_normalized
        
        dt_scaled = torch.sigmoid(self.dt.float()) * 1.0
        omega = self.frequencies.float() * spectral_freq_scale * dt_scaled
        rot = torch.complex(torch.cos(omega), torch.sin(omega))
        
        A = decay * rot
        
        # Build frequency-band balanced kernel
        ones = torch.ones(1, 1, self.n_frequencies, device=device, dtype=A.dtype)
        A_expanded = A.view(1, 1, self.n_frequencies).expand(B, T, self.n_frequencies)
        
        band_scale = torch.zeros(B, T, self.n_frequencies, device=device)
        for i in range(self.n_bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < self.n_bands - 1 else self.n_frequencies
            band_scale[:, :, start_idx:end_idx] = band_strengths[:, :, i:i+1]
        
        A_expanded = A_expanded * band_scale.to(A_expanded.dtype)
        
        # Cumulative product
        powers = torch.cumprod(
            torch.cat([ones.expand(B, -1, -1), A_expanded[:, :-1, :]], dim=1),  
            dim=1
        )
        
        # VECTORIZED magnitude clamping
        powers_magnitude = torch.abs(powers)
        scale_factor = torch.tanh(powers_magnitude / 10.0) * 10.0 / (powers_magnitude + 1e-8)
        powers = powers * scale_factor
        
        powers = powers * temporal_strength.float()
        kernel = powers * rot.view(1, 1, self.n_frequencies)
        
        # FFT-based causal convolution
        n_fft = 2 * T
        U_f = torch.fft.fft(u, n=n_fft, dim=1)
        K_f = torch.fft.fft(kernel, n=n_fft, dim=1)
        Y_f = U_f * K_f
        y = torch.fft.ifft(Y_f, n=n_fft, dim=1)[:, :T, :]
        
        # Convert back
        spectral_out = torch.cat([y.real, y.imag], dim=-1).to(original_dtype)
        
        y_proj = self.from_spectral(spectral_out)
        y_proj = y_proj * self.output_scale
        
        # Gates with temperature
        gate_values = torch.sigmoid(self.gate(x_norm) / self.gate_temperature)
        y_gated = (y_proj * gate_values + concept_preserved) / 2.0
        y_gated = self.dropout(y_gated)
        
        x = residual + y_gated
        x = x + self.ffn(self.norm_ffn(x))
        
        return x

# ==========================================
# OPTION 2: Sliding Window Attention (FIXED v2)
# ==========================================
class SlidingWindowAttention(nn.Module):
    """Efficient long-range attention via strided processing"""
    def __init__(self, d_model, n_heads, rope_emb, stride=4):
        super().__init__()
        self.stride = stride
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_emb = rope_emb
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, D = x.shape
        residual = x
        x_norm = self.norm(x)
        
        # Handle small sequences - fallback to standard attention
        if T < self.stride * 2:
            qkv = self.qkv(x_norm)
            q, k, v = qkv.reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            
            cos, sin = self.rope_emb(v, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=dropout if self.training else 0.0
            )
            
            out = out.transpose(1, 2).reshape(B, T, D)
            out = self.out_proj(out)
            out = self.dropout(out)
            return residual + out
        
        # Strided sampling for long sequences
        x_strided = x_norm[:, ::self.stride, :]  # [B, T/stride, D]
        T_strided = x_strided.size(1)  # Use actual strided length
        
        # Safety check - should always have at least 1 token after stride
        if T_strided == 0:
            x_strided = x_norm[:, :1, :]
            T_strided = 1
        
        qkv = self.qkv(x_strided)
        q, k, v = qkv.reshape(B, T_strided, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        cos, sin = self.rope_emb(v, seq_len=T_strided)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Full attention on strided sequence (much cheaper)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=dropout if self.training else 0.0
        )
        
        out = out.transpose(1, 2).reshape(B, T_strided, D)
        out = self.out_proj(out)
        
        # Interpolate back to full sequence
        if T_strided == 1 and T > 1:
            # Special case: broadcast single position to all
            out = out.expand(-1, T, -1)
        else:
            out = F.interpolate(
                out.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False if T_strided > 1 else True
            ).transpose(1, 2)
        
        out = self.dropout(out)
        return residual + out

# ==========================================
# OPTION 3: Memory Tokens
# ==========================================
class MemoryTokenLayer(nn.Module):
    """Learnable memory tokens that compress global context"""
    def __init__(self, d_model, n_heads, n_memory_tokens, rope_emb):
        super().__init__()
        self.n_memory = n_memory_tokens
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_emb = rope_emb
        
        # Learnable memory tokens
        self.memory_tokens = nn.Parameter(torch.randn(1, n_memory_tokens, d_model) * 0.02)
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.norm_memory = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, D = x.shape
        residual = x
        x_norm = self.norm(x)
        
        # Prepend memory tokens
        memory = self.memory_tokens.expand(B, -1, -1)
        memory = self.norm_memory(memory)
        x_with_mem = torch.cat([memory, x_norm], dim=1)  # [B, M+T, D]
        
        qkv = self.qkv(x_with_mem)
        q, k, v = qkv.reshape(B, self.n_memory + T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        cos, sin = self.rope_emb(v, seq_len=self.n_memory + T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Memory tokens can attend bidirectionally to capture global context
        # Regular tokens attend causally
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,  # Still causal overall
            dropout_p=dropout if self.training else 0.0
        )
        
        out = out.transpose(1, 2).reshape(B, self.n_memory + T, D)
        out = self.out_proj(out)
        
        # Remove memory tokens, keep only sequence output
        out = out[:, self.n_memory:, :]
        out = self.dropout(out)
        
        return residual + out

# --- Standard Causal Attention Block ---
class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, rope_emb):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_emb = rope_emb
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), 
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        residual = x
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = self.rope_emb(v, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True,
            dropout_p=dropout if self.training else 0.0
        )
        
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        x = residual + out
        return x + self.ffn(self.norm_ffn(x))

# --- Bidirectional Attention Block (for BERT) ---
class BidirectionalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, rope_emb):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_emb = rope_emb
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), 
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        residual = x
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = self.rope_emb(v, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=False,
            dropout_p=dropout if self.training else 0.0
        )
        
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        x = residual + out
        return x + self.ffn(self.norm_ffn(x))

# ==========================================
# Enhanced Hierarchical Transformer
# ==========================================
class EnhancedHierarchicalTransformer(nn.Module):
    def __init__(self, tokens_to_predict=5, vocab_size=50304):
        super().__init__()
        self.tokens_to_predict = tokens_to_predict
        self.vocab_size = vocab_size
        
        self.tok_embeddings = nn.Embedding(vocab_size + 1, n_embd)
        self.rope = RotaryEmbedding(n_embd // n_head)
        
        # ARCHITECTURE: More spectral + three long-range mechanisms
        # [Spectral, Spectral, Memory, Spectral, Sliding, Spectral, Attention, Memory]
        
        self.layers = nn.ModuleList([
            EnhancedSpectralBlock(n_embd, n_frequencies, block_size),  # 0
            EnhancedSpectralBlock(n_embd, n_frequencies, block_size),  # 1
            MemoryTokenLayer(n_embd, n_head, n_memory_tokens, self.rope),  # 2 - Memory
            EnhancedSpectralBlock(n_embd, n_frequencies, block_size),  # 3
            SlidingWindowAttention(n_embd, n_head, self.rope, stride=window_stride),  # 4 - Sliding
            EnhancedSpectralBlock(n_embd, n_frequencies, block_size),  # 5
            CausalAttentionBlock(n_embd, n_head, self.rope),  # 6
            MemoryTokenLayer(n_embd, n_head, n_memory_tokens, self.rope),  # 7 - Memory
        ])
        
        # Separate BERT encoder
        self.bert_layers = nn.ModuleList([
            BidirectionalAttentionBlock(n_embd, n_head, self.rope) 
            for _ in range(n_layer)
        ])
        
        self.norm = nn.LayerNorm(n_embd)
        self.bert_norm = nn.LayerNorm(n_embd)
        
        # Two heads + memory auxiliary head
        self.spectral_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.main_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.memory_head = nn.Linear(n_embd, vocab_size, bias=False)  # For memory token loss
        
        self.bert_head = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, vocab_size)
        )
        
        # Weight tying
        self.tok_embeddings.weight = nn.Parameter(
            torch.cat([self.main_head.weight, torch.randn(1, n_embd) * 0.02], dim=0)
        )
        
        # *** ADAPTIVE LOSS WEIGHTING ***
        self.adaptive_loss = AdaptiveWeightedLoss(num_losses=3)

        # Print architecture info
        print(f"\n{'='*80}")
        print(f"Enhanced Hierarchical Transformer with Adaptive Loss Weighting")
        print(f"{'='*80}")
        print(f"Architecture ({len(self.layers)} layers):")
        print(f"  Layer 0-1: Enhanced Spectral (n_freq={n_frequencies})")
        print(f"  Layer 2: Memory Tokens (n={n_memory_tokens})")
        print(f"  Layer 3: Enhanced Spectral")
        print(f"  Layer 4: Sliding Window Attention (stride={window_stride})")
        print(f"  Layer 5: Enhanced Spectral")
        print(f"  Layer 6: Standard Causal Attention")
        print(f"  Layer 7: Memory Tokens")
        print(f"\n  BERT Encoder: {n_layer} bidirectional layers")
        print(f"\n  Loss Balancing: Adaptive Uncertainty Weighting (Learnable)")
        print(f"  Long-Range Mechanisms: Spectral + Memory + Sliding Window")
        print(f"{'='*80}\n")

    def forward(self, idx, targets=None, bert_mode=False, bert_labels=None):
        B, T = idx.shape
        x = self.tok_embeddings(idx)

        if bert_mode:
            for layer in self.bert_layers:
                x = layer(x)
            
            x = self.bert_norm(x)
            logits_bert = self.bert_head(x)
            
            loss = None
            if bert_labels is not None:
                loss = F.cross_entropy(
                    logits_bert.view(B * T, self.vocab_size),
                    bert_labels.view(B * T),
                    ignore_index=-100
                )
            
            return None, None, None, loss
        
        else:
            # Forward through enhanced layers
            for layer in self.layers:
                x = layer(x)
            
            x = self.norm(x)
            logits_main = self.main_head(x)
            logits_spec = self.spectral_head(x)
            logits_memory = self.memory_head(x)

            loss = None
            if targets is not None:
                # Main loss
                logits_main_flat = logits_main.view(B*T, self.vocab_size)
                targets_flat = targets.view(B*T)
                loss_main = F.cross_entropy(
                    logits_main_flat, targets_flat, 
                    label_smoothing=label_smoothing
                )

                # Spectral multi-position loss
                loss_spec = 0
                if self.tokens_to_predict > 1:
                    for i in range(1, self.tokens_to_predict):
                        if i < T:
                            future_targets = targets[:, i:]
                            logits_trimmed = logits_spec[:, :-i]
                            
                            pos_loss = F.cross_entropy(
                                logits_trimmed.reshape(-1, self.vocab_size),
                                future_targets.reshape(-1),
                                label_smoothing=label_smoothing
                            )
                            
                            loss_spec += pos_loss * (0.5 ** i)
                    
                    loss_spec = loss_spec / (self.tokens_to_predict - 1)
                
                # Memory auxiliary loss
                loss_memory = F.cross_entropy(
                    logits_memory.view(B*T, self.vocab_size),
                    targets_flat,
                    label_smoothing=label_smoothing
                )
                
                # *** USE ADAPTIVE WEIGHTING ***
                losses = {
                    'auto': loss_main,
                    'spectral': loss_spec,
                    'memory': loss_memory
                }
                loss = self.adaptive_loss(losses)

            return logits_main, logits_spec, logits_memory, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.5):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits_main, _, _, _ = self(idx_cond, bert_mode=False)
            logits = logits_main[:, -1, :] / temperature
            
            # Enhanced repetition penalty for generation
            if repetition_penalty > 1.0:
                context_window = idx_cond[:, -repetition_context_size:]
                for b in range(idx.size(0)):
                    token_counts = {}
                    for token in context_window[b].tolist():
                        token_counts[token] = token_counts.get(token, 0) + 1
                    
                    for token, count in token_counts.items():
                        if token < self.vocab_size:
                            token_penalty = repetition_penalty + (0.1 * (count - 1))
                            if logits[b, token] > 0:
                                logits[b, token] /= token_penalty
                            else:
                                logits[b, token] *= token_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# 4. Training Utilities
# ==========================================

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, mask_for_bert=False)
            _, _, _, loss = model(X, Y, bert_mode=False)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < 100: return learning_rate * it / 100
    if it > lr_decay_iters: return min_lr
    decay_ratio = (it - 100) / (lr_decay_iters - 100)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ==========================================
# 5. Gradient Monitoring Utility
# ==========================================
def compute_grad_norm(model):
    """Compute the total gradient norm across all parameters"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# ==========================================
# 6. Main Execution
# ==========================================

if __name__ == '__main__':
    generator = EnhancedHierarchicalTransformer(
        tokens_to_predict=tokens_to_predict, 
        vocab_size=vocab_size
    ).to(device)
    
    # Initialize basic training variables
    iter_num = 0
    best_val_loss = 1e9
    optimizer = torch.optim.AdamW(
        generator.parameters(), 
        lr=learning_rate, 
        betas=(beta1, beta2), 
        weight_decay=weight_decay
    )

    # CHECKPOINT HANDLING
    if os.path.exists(ckpt_path):
        print(f"\n>>> Checkpoint found at {ckpt_path}!")
        choice = input(">>> Type 'c' for CHAT MODE or 't' to CONTINUE TRAINING: ").lower().strip()
        
        print(">>> Loading checkpoint...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Load model weights
        keys = generator.load_state_dict(checkpoint['model'], strict=False)
        if keys.missing_keys:
            print(f"Note: Loaded with missing keys: {keys.missing_keys}")
        
        if choice == 'c':
            # --- CHAT MODE ---
            generator.eval()
            print(">>> Model loaded in EVAL mode. Type 'exit' to quit.\n")
            
            while True:
                user_input = input("User: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input.strip():
                    continue

                ids = enc.encode(user_input)
                idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
                
                generated_idx = generator.generate(
                    idx, 
                    max_new_tokens=200, 
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=repetition_penalty_generate
                )
                
                new_tokens = generated_idx[0].tolist()[len(ids):]
                response = enc.decode(new_tokens)
                print(f"Bot: {response}\n")
                
            print("Exiting chat mode.")
            exit() 
        
        else:
            # --- CONTINUE TRAINING ---
            print(">>> Resuming training from checkpoint...")
            optimizer.load_state_dict(checkpoint['optimizer'])
            iter_num = checkpoint.get('iter_num', 0)
            best_val_loss = checkpoint.get('best_val_loss', 1e9)
            print(f">>> Resuming from iteration {iter_num}")

    else:
        print(">>> No checkpoint found. Starting TRAINING from scratch...")

    # --- TRAINING LOOP SETUP ---
    gen_params = sum(p.numel() for p in generator.parameters())/1e6
    print(f"Generator params: {gen_params:.2f}M\n")
    
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    while iter_num <= max_iters:

        if iter_num % eval_interval == 0:
            generator.eval() # Ensure eval mode for validation
            with torch.no_grad():
                losses = estimate_loss(generator)
            
            print(f"step {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            
            # Sample generation
            context = torch.zeros((1, 1), dtype=torch.long, device=device) 
            gen_tokens = generator.generate(
                context, 
                max_new_tokens=50,
                temperature=0.8,
                top_k=40, 
                top_p=0.9,
                repetition_penalty=repetition_penalty_generate
            )[0].tolist()
            valid_tokens = [t for t in gen_tokens if t < enc.n_vocab]
            print(f"Gen: {enc.decode(valid_tokens)}")
            print("-" * 60)

            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint_data = {
                        'model': generator.state_state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    torch.save(checkpoint_data, ckpt_path)
            
            generator.train() # Switch back to train mode

        # Update learning rate based on current iter_num
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups: 
            param_group['lr'] = lr
        
        xb, yb = get_batch('train', mask_for_bert=False)
        is_bert_step = (bert_every_n_steps > 0) and (iter_num % bert_every_n_steps == 0)
        
        if is_bert_step:
            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=ptdtype):
                _, _, _, loss_auto = generator(xb, yb, bert_mode=False)
                xb_masked, bert_labels, _ = get_batch('train', mask_for_bert=True)
                _, _, _, loss_bert = generator(xb_masked, bert_labels=bert_labels, bert_mode=True)
                total_loss = loss_auto + loss_bert 
            
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = compute_grad_norm(generator)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            optimizer.step()
            
            if iter_num % log_interval == 0:
                dt = time.time() - t0
                t0 = time.time()
                weights = generator.adaptive_loss.get_weights()
                print(f"iter {iter_num}: loss_AUTO {loss_auto.item():.4f} + loss_BERT {loss_bert.item():.4f} | "
                      f"weights A:{weights['w_auto']:.3f} S:{weights['w_spectral']:.3f} "
                      f"M:{weights['w_memory']:.3f} | grad_norm {grad_norm:.4f} | time {dt*1000:.2f}ms")
            
        else:
            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=ptdtype):
                _, _, _, loss_auto = generator(xb, yb, bert_mode=False)

            optimizer.zero_grad(set_to_none=True)
            loss_auto.backward()
            grad_norm = compute_grad_norm(generator)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            optimizer.step()

            if iter_num % log_interval == 0:
                dt = time.time() - t0
                t0 = time.time()
                weights = generator.adaptive_loss.get_weights()
                print(f"iter {iter_num}: loss_AUTO {loss_auto.item():.4f} | "
                      f"weights A:{weights['w_auto']:.3f} S:{weights['w_spectral']:.3f} "
                      f"M:{weights['w_memory']:.3f} | grad_norm {grad_norm:.4f} | time {dt*1000:.2f}ms")

        iter_num += 1

    # Final Save
    print("Training finished. Saving final checkpoint...")
    torch.save({
        'model': generator.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }, ckpt_path)
    print(f"Final checkpoint saved to {ckpt_path}")