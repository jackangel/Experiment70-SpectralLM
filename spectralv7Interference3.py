import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import tiktoken
import numpy as np

# ==========================================
# 1. Hyperparameters & Configuration
# ==========================================
# I/O
out_dir = 'out'
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
eval_interval = 1000
log_interval = 100
eval_iters = 50
always_save_checkpoint = True 

# Data
batch_size = 4       
block_size = 512     
dataset = 'input.txt' 

# Model Architecture
n_layer = 6          
n_head = 8
n_embd = 512         
dropout = 0.1
label_smoothing = 0.1

# Spectral Settings
n_frequencies = 512       
spectral_decay_init = -3.0  
spectral_freq_scale = 0.1   
tokens_to_predict = 7
spectral_loss_weight = 0.15

# CNN Helper Settings
cnn_base_channels = 256
cnn_loss_weight = 0.1

# BERT-style Training Settings
bert_every_n_steps = 5
bert_mask_prob = 0.15
bert_loss_weight = 0.1

# Optimizer
learning_rate = 4e-4  
max_iters = 50000
lr_decay_iters = 5000
min_lr = 4e-5
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95 
grad_clip = 1.0

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)

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
# Spectral State Space Block
# ==========================================
class ImprovedSpectralBlock(nn.Module):
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
        
        # Frequency grouping
        self.n_bands = 4
        self.band_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            ) for _ in range(self.n_bands)
        ])
        
        # State space parameters
        freq_groups = torch.linspace(spectral_decay_init, -0.001, n_frequencies)
        freq_groups = freq_groups + torch.randn(n_frequencies) * 0.1
        self.log_decay = nn.Parameter(freq_groups)
        
        freq_init = torch.zeros(n_frequencies)
        for i in range(n_frequencies):
            freq_init[i] = (i / n_frequencies) * 2.0 - 1.0
        self.frequencies = nn.Parameter(freq_init * 0.2)
        
        # Temporal context gate
        self.temporal_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.dt = nn.Parameter(torch.ones(n_frequencies) * 0.5)
        
        # Concept preservation
        self.concept_proj = nn.Linear(d_model, d_model)
        self.concept_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.gate = nn.Linear(d_model, d_model)
        self.from_spectral = nn.Linear(n_frequencies * 2, d_model, bias=False)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.5)
        
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), 
            nn.GELU(), 
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
        
        # Forget gates
        forget_gates = self.freq_forget_gate(x_norm)
        temporal_strength = self.temporal_gate(x_norm)
        
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
        freq_importance_normalized = F.softmax(self.freq_importance.float(), dim=0) * self.n_frequencies
        decay = decay_base * freq_importance_normalized
        
        dt_scaled = torch.sigmoid(self.dt.float()) * 2.0
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
        
        powers = torch.cumprod(
            torch.cat([ones.expand(B, -1, -1), A_expanded[:, :-1, :]], dim=1),  
            dim=1
        )
        
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
        
        # Combine with concept preservation
        gate_values = torch.sigmoid(self.gate(x_norm))
        y_gated = (y_proj * gate_values + concept_preserved) / 2.0
        y_gated = self.dropout(y_gated)
        
        x = residual + y_gated
        x = x + self.ffn(self.norm_ffn(x))
        
        return x

# ==========================================
# NEW: Causal Hierarchical Block
# ==========================================
class CausalHierarchicalBlock(nn.Module):
    def __init__(self, d_model, compression_factor=4, kernel_size=15):
        super().__init__()
        self.compression_factor = compression_factor
        
        # 1. Causal Downsampling (Strided Convolution)
        self.downsample = nn.Conv1d(
            d_model, d_model, 
            kernel_size=compression_factor, 
            stride=compression_factor
        )
        
        # 2. Low-Frequency Processing (The "Spectral" View)
        self.low_freq_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=0, groups=d_model), # Depthwise
            nn.GroupNorm(min(32, d_model), d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1) # Pointwise
        )
        self.kernel_size = kernel_size
        
        # 3. Mixing Gate
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        
        # ------------------------------------------------
        # Step 1: Causal Padding for Downsampling
        # ------------------------------------------------
        remainder = T % self.compression_factor
        pad_amt = (self.compression_factor - remainder) % self.compression_factor
        
        # Transpose for Conv: [B, D, T]
        x_conv = x.transpose(1, 2)
        
        # Apply left padding
        x_padded = F.pad(x_conv, (pad_amt, 0))
        
        # ------------------------------------------------
        # Step 2: Downsample
        # ------------------------------------------------
        x_compressed = self.downsample(x_padded)
        
        # ------------------------------------------------
        # Step 3: Process Long-Range Patterns
        # ------------------------------------------------
        proc_padding = self.kernel_size - 1
        x_comp_padded = F.pad(x_compressed, (proc_padding, 0))
        x_processed = self.low_freq_conv(x_comp_padded)
        
        # ------------------------------------------------
        # Step 4: Upsample
        # ------------------------------------------------
        x_upsampled = torch.repeat_interleave(x_processed, self.compression_factor, dim=2)
        
        # Trim
        if x_upsampled.shape[2] > T:
            x_upsampled = x_upsampled[:, :, -T:]
        elif x_upsampled.shape[2] < T:
            x_upsampled = F.pad(x_upsampled, (T - x_upsampled.shape[2], 0))
            
        # Back to [B, T, D]
        x_out = x_upsampled.transpose(1, 2)
        
        # ------------------------------------------------
        # Step 5: Gated Residual
        # ------------------------------------------------
        gate_val = self.gate(x_out)
        return x + (x_out * gate_val)

# ==========================================
# CAUSAL Long-Range CNN Helper Network (MODIFIED)
# ==========================================
class LongRangeCNNHelper(nn.Module):
    def __init__(self, d_model, vocab_size, base_channels=256):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # ============================================
        # Branch 1: CAUSAL Dilated Convolution Tower
        # ============================================
        self.dilated_tower = nn.ModuleList()
        self.dilated_paddings = [] 
        self.dilations = [1, 2, 4, 8, 16] 
        
        for i, dilation in enumerate(self.dilations):
            is_first = (i == 0)
            in_ch = d_model if is_first else base_channels
            kernel_size = 3
            
            # CAUSAL padding: only pad on the LEFT (past)
            padding = dilation * (kernel_size - 1)
            self.dilated_paddings.append(padding)
            
            self.dilated_tower.append(nn.ModuleDict({
                'conv': nn.Conv1d(
                    in_ch, base_channels, 
                    kernel_size=kernel_size, 
                    dilation=dilation,
                    padding=0 
                ),
                'norm': nn.GroupNorm(min(32, base_channels), base_channels),
                'activation': nn.GELU(),
            }))
        
        # ============================================
        # Branch 2: Causal Hierarchical Spectral Block (NEW)
        # ============================================
        self.hierarchy_block = CausalHierarchicalBlock(d_model, compression_factor=4, kernel_size=15)
        # We need to project the hierarchy output (d_model) down to base_channels for fusion
        self.hierarchy_proj = nn.Conv1d(d_model, base_channels, kernel_size=1)
        
        # ============================================
        # Branch 3: CAUSAL Large Kernel
        # ============================================
        self.kernel_size_large = 31 
        self.large_kernel_padding1 = self.kernel_size_large - 1
        
        self.large_kernel_branch = nn.ModuleDict({
            'conv1': nn.Conv1d(d_model, base_channels, kernel_size=self.kernel_size_large, padding=0),
            'norm1': nn.GroupNorm(min(32, base_channels), base_channels),
        })
        
        # ============================================
        # Feature Fusion
        # ============================================
        # Channels = Tower outputs (len(dilations)) + Large Kernel (1) + Hierarchy (1)
        total_channels = base_channels * (len(self.dilations) + 1 + 1)
        
        self.fusion = nn.Sequential(
            nn.Conv1d(total_channels, base_channels, kernel_size=1),
            nn.GroupNorm(min(32, base_channels), base_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ============================================
        # Output Projection
        # ============================================
        self.to_embedding_space = nn.Sequential(
            nn.Conv1d(base_channels, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # ============================================
        # Cross-Modal Fusion Gate
        # ============================================
        self.cross_attn_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.cnn_strength = nn.Parameter(torch.tensor(-4.5))

    def causal_pad(self, x, padding):
        """Apply causal padding (only on the left/past)"""
        if padding == 0:
            return x
        return F.pad(x, (padding, 0))

    def forward(self, x, return_features=False):
        B, T, D = x.shape
        
        # Transpose for conv1d: [B, D, T]
        x_conv = x.transpose(1, 2)
        
        all_features = []
        
        # 1. Process Dilated Tower
        tower_out = x_conv
        for i, layer in enumerate(self.dilated_tower):
            padding = self.dilated_paddings[i]
            padded = self.causal_pad(tower_out, padding)
            
            conv_out = layer['conv'](padded)
            conv_out = layer['norm'](conv_out)
            conv_out = layer['activation'](conv_out)
            
            if conv_out.shape[2] > T:
                conv_out = conv_out[:, :, :T]
            
            if tower_out.shape[1] == conv_out.shape[1]:
                conv_out = conv_out + tower_out
                
            all_features.append(conv_out)
            tower_out = conv_out
            
        # 2. Process Large Kernel
        padded = self.causal_pad(x_conv, self.large_kernel_padding1)
        lk_out = self.large_kernel_branch['conv1'](padded)
        lk_out = self.large_kernel_branch['norm1'](lk_out)
        lk_out = F.gelu(lk_out)
        if lk_out.shape[2] > T:
            lk_out = lk_out[:, :, :T]
            
        all_features.append(lk_out)

        # 3. Process Hierarchy (NEW)
        # Input to hierarchy is [B, T, D]
        h_out = self.hierarchy_block(x) 
        # Transpose to [B, D, T] for projection
        h_out_conv = h_out.transpose(1, 2)
        # Project to base_channels
        h_out_proj = self.hierarchy_proj(h_out_conv)
        
        all_features.append(h_out_proj)
        
        # 4. Fuse
        concatenated = torch.cat(all_features, dim=1)
        fused = self.fusion(concatenated)
        
        # 5. Project
        features_conv = self.to_embedding_space(fused)
        features = features_conv.transpose(1, 2) # [B, T, D]
        features = self.final_norm(features)
        
        logits = self.head(features)
        
        if return_features:
            return logits, features
        return logits
    
    def get_cross_attention_signal(self, main_features, cnn_features):
        combined = torch.cat([main_features, cnn_features], dim=-1)
        gate = self.cross_attn_gate(combined)
        strength = torch.sigmoid(self.cnn_strength)
        modulated = main_features + gate * cnn_features * strength
        return modulated

    def get_receptive_field_info(self):
        rf_dilated = 1
        for d in self.dilations:
            rf_dilated += (3 - 1) * d
            
        return {
            'dilated_tower': rf_dilated,
            'large_kernel': self.kernel_size_large,
            'hierarchy': '4x Compression * 15 Kernel = 60 (Effective)',
            'max_receptive_field': max(rf_dilated, self.kernel_size_large, 60),
            'info': 'Fixed Causal Version with Hierarchy'
        }

# --- Standard Attention Block ---
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

# --- Bidirectional Attention Block ---
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
# Hierarchical Transformer with Long-Range CNN
# ==========================================
class HierarchicalTransformer(nn.Module):
    def __init__(self, tokens_to_predict=5, vocab_size=50304):
        super().__init__()
        self.tokens_to_predict = tokens_to_predict
        self.vocab_size = vocab_size
        
        self.tok_embeddings = nn.Embedding(vocab_size + 1, n_embd)
        self.rope = RotaryEmbedding(n_embd // n_head)
        
        # Hierarchical design
        n_spectral = n_layer // 2
        n_attention = n_layer - n_spectral
        
        # Main pathway
        self.layers = nn.ModuleList([
            ImprovedSpectralBlock(n_embd, n_frequencies, block_size) 
            for _ in range(n_spectral)
        ])
        
        self.layers.extend([
            CausalAttentionBlock(n_embd, n_head, self.rope) 
            for _ in range(n_attention)
        ])
        
        # Long-Range CNN Helper Network
        self.cnn_helper = LongRangeCNNHelper(
            n_embd, 
            vocab_size,
            base_channels=cnn_base_channels
        )
        
        # Separate BERT encoder
        self.bert_layers = nn.ModuleList([
            BidirectionalAttentionBlock(n_embd, n_head, self.rope) 
            for _ in range(n_layer)
        ])
        
        self.norm = nn.LayerNorm(n_embd)
        self.bert_norm = nn.LayerNorm(n_embd)
        
        # Three heads
        self.spectral_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.main_head = nn.Linear(n_embd, vocab_size, bias=False)
        
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

        # Print architecture info
        rf_info = self.cnn_helper.get_receptive_field_info()
        
        print(f"\n{'='*70}")
        print(f"Hierarchical Transformer with Long-Range CNN Helper")
        print(f"{'='*70}")
        print(f"Architecture:")
        print(f"  Main Pathway:")
        print(f"    - Lower {n_spectral} layers: State Space Spectral")
        print(f"    - Upper {n_attention} layers: Causal Attention")
        print(f"  Long-Range CNN Helper:")
        print(f"    - Dilated Tower: {rf_info['dilated_tower']} token receptive field")
        print(f"    - Large Kernel Branch: {rf_info['large_kernel']} tokens")
        print(f"    - Hierarchy Block: {rf_info['hierarchy']}")
        print(f"    - Max Receptive Field: {rf_info['max_receptive_field']} tokens")
        print(f"  BERT Encoder: {n_layer} bidirectional layers")
        print(f"\nObjectives:")
        print(f"  - Next Token Prediction (main)")
        print(f"  - Multi-Position Prediction (spectral, 1-{tokens_to_predict} steps)")
        print(f"  - CNN Long-Range Pattern Prediction")
        print(f"  - BERT Masked LM (auxiliary)")
        print(f"{'='*70}\n")

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
            # Get CNN features early in the pipeline
            cnn_logits, cnn_features = self.cnn_helper(x, return_features=True)
            
            # Forward through main pathway
            for i, layer in enumerate(self.layers):
                x = layer(x)
                
                # Inject CNN features at middle layers
                if i == len(self.layers) // 2:
                    x = self.cnn_helper.get_cross_attention_signal(x, cnn_features)
            
            x = self.norm(x)
            logits_main = self.main_head(x)
            logits_spec = self.spectral_head(x)

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
                
                if self.tokens_to_predict > 1:
                    loss_spec = loss_spec / (self.tokens_to_predict - 1)
                
                # CNN auxiliary loss
                loss_cnn = F.cross_entropy(
                    cnn_logits.view(B*T, self.vocab_size),
                    targets_flat,
                    label_smoothing=label_smoothing
                )
                
                # Combine all losses
                loss = (loss_main + 
                       (loss_spec * spectral_loss_weight) + 
                       (loss_cnn * cnn_loss_weight))

            return logits_main, logits_spec, cnn_logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits_main, _, _, _ = self(idx_cond, bert_mode=False)
            logits = logits_main[:, -1, :] / temperature
            
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
            if torch.isnan(probs).any(): 
                probs = torch.ones_like(probs) / self.vocab_size
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
# 5. Main Execution
# ==========================================

if __name__ == '__main__':
    generator = HierarchicalTransformer(
        tokens_to_predict=tokens_to_predict, 
        vocab_size=vocab_size
    ).to(device)
    
    if os.path.exists(ckpt_path):
        print(f"\n>>> Checkpoint found at {ckpt_path}!")
        print(">>> Loading model and entering CHAT MODE...")
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        generator.load_state_dict(checkpoint['model'])
        generator.eval()
        
        print(">>> Model loaded. Type 'exit' to quit.\n")
        
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
                temperature=0.7,
                top_k=40,
                top_p=0.92
            )
            
            new_tokens = generated_idx[0].tolist()[len(ids):]
            response = enc.decode(new_tokens)
            
            print(f"Bot: {response}\n")
            
        print("Exiting chat mode.")
        exit() 

    print(">>> No checkpoint found. Starting TRAINING...")

    optimizer = torch.optim.AdamW(
        generator.parameters(), 
        lr=learning_rate, 
        betas=(beta1, beta2), 
        weight_decay=weight_decay
    )

    gen_params = sum(p.numel() for p in generator.parameters())/1e6
    print(f"Generator params: {gen_params:.2f}M\n")
    
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True

    iter_num = 0
    best_val_loss = 1e9
    
    t0 = time.time()
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    while iter_num <= max_iters:

        if iter_num % eval_interval == 0:
            losses = estimate_loss(generator)
            current_val = losses['val'].item()
            
            print(f"step {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            
            context = torch.zeros((1, 1), dtype=torch.long, device=device) 
            gen_tokens = generator.generate(
                context, 
                max_new_tokens=50,
                temperature=0.8,
                top_k=50, 
                top_p=0.9
            )[0].tolist()
            valid_tokens = [t for t in gen_tokens if t < enc.n_vocab]
            print(f"Gen: {enc.decode(valid_tokens)}")
            print("-" * 60)

            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    torch.save({
                        'model': generator.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                    }, ckpt_path)

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
                total_loss = loss_auto + (bert_loss_weight * loss_bert)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            optimizer.step()
            
            if iter_num % log_interval == 0:
                dt = time.time() - t0
                t0 = time.time()
                print(f"iter {iter_num}: loss_AUTO {loss_auto.item():.4f} + loss_BERT {loss_bert.item():.4f} | time {dt*1000:.2f}ms")
            
        else:
            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=ptdtype):
                _, _, _, loss_auto = generator(xb, yb, bert_mode=False)

            optimizer.zero_grad()
            loss_auto.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            optimizer.step()

            if iter_num % log_interval == 0:
                dt = time.time() - t0
                t0 = time.time()
                print(f"iter {iter_num}: loss_AUTO {loss_auto.item():.4f} | time {dt*1000:.2f}ms")

        iter_num += 1

    print("Training finished. Saving final checkpoint...")
    torch.save({
        'model': generator.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
    }, ckpt_path)
    print(f"Final checkpoint saved to {ckpt_path}")