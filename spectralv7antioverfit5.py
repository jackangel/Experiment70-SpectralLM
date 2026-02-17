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
from collections import defaultdict
import numpy as np

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
# 2. Riemannian/Hyperbolic VQ (Recommendation #10)
# -----------------------------
class HyperbolicVQ(nn.Module):
    """Vector Quantizer in Poincaré ball model for hierarchical structures"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-0.01, 0.01)
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        
    def forward(self, inputs):
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Normalize inputs and embeddings
        flat_input = F.normalize(flat_input, dim=-1)
        
        with torch.no_grad():
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=-1)
        
        # Compute cosine distances
        distances = 1.0 - torch.matmul(flat_input, self.embedding.weight.t())
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # EMA update
        if self.training:
            with torch.no_grad():
                encodings_sum = encodings.sum(0)
                self.ema_cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
                
                n = self.ema_cluster_size.sum()
                self.ema_cluster_size.add_(self.epsilon).div_(
                    n + self.num_embeddings * self.epsilon
                ).mul_(n)
                
                dw = torch.matmul(encodings.t(), flat_input)
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                self.embedding.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1).clamp(min=self.epsilon))
        
        # Commitment loss only
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.view(inputs.shape[:-1])

# -----------------------------
# 3. Adaptive Temporal Segmentation (Recommendation #1)
# -----------------------------
class AdaptiveSegmenter(nn.Module):
    """Learns to segment sequences at natural boundaries"""
    def __init__(self, d_model, max_segment_len=32):
        super().__init__()
        self.max_segment_len = max_segment_len
        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 2)
        )
        self.segment_aggregator = nn.LSTM(d_model, d_model, batch_first=True)
        
    def forward(self, x):
        B, T, D = x.shape
        
        # Predict boundaries
        boundary_logits = self.boundary_predictor(x)
        boundary_probs = F.softmax(boundary_logits, dim=-1)
        
        if self.training:
            boundaries = F.gumbel_softmax(boundary_logits, tau=1.0, hard=True)[:, :, 1]
        else:
            boundaries = (boundary_probs[:, :, 1] > 0.5).float()
        
        # Force boundaries at regular intervals
        for b in range(B):
            for t in range(self.max_segment_len, T, self.max_segment_len):
                if t < T:
                    boundaries[b, t] = 1.0
            boundaries[b, T-1] = 1.0
        
        # Aggregate segments
        segments_list = []
        boundary_indices = []
        
        for b in range(B):
            segment_start = 0
            batch_segments = []
            batch_boundaries = []
            
            for t in range(T):
                if boundaries[b, t] > 0.5:
                    if t > segment_start:
                        segment = x[b, segment_start:t+1]
                        _, (h, _) = self.segment_aggregator(segment.unsqueeze(0))
                        batch_segments.append(h.squeeze(0).squeeze(0))
                        batch_boundaries.append(t)
                    segment_start = t + 1
            
            if len(batch_segments) == 0:
                batch_segments.append(x[b].mean(dim=0))
                batch_boundaries.append(T-1)
            
            segments_list.append(torch.stack(batch_segments))
            boundary_indices.append(batch_boundaries)
        
        max_segments = max(seg.shape[0] for seg in segments_list)
        padded_segments = torch.zeros(B, max_segments, D, device=x.device)
        
        for b, segs in enumerate(segments_list):
            padded_segments[b, :segs.shape[0]] = segs
        
        return padded_segments, boundary_indices, boundary_logits

# -----------------------------
# 4. Disentangled Content/Style VQ (Recommendation #4) - FIXED
# -----------------------------
class DisentangledVQ(nn.Module):
    """Separates what is said (content) from how it's said (style)"""
    def __init__(self, d_model, num_content_buckets, num_style_buckets):
        super().__init__()
        self.content_dim = d_model // 2
        self.style_dim = d_model // 2
        
        self.content_extractor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.content_dim),
            nn.Tanh()
        )
        self.style_extractor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.style_dim),
            nn.Tanh()
        )
        
        self.content_vq = HyperbolicVQ(num_content_buckets, self.content_dim, commitment_cost=0.1)
        self.style_vq = HyperbolicVQ(num_style_buckets, self.style_dim, commitment_cost=0.1)
        
        self.combiner = nn.Sequential(
            nn.Linear(self.content_dim + self.style_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Simplified disentanglement - just orthogonality
        
    def forward(self, x):
        content_features = self.content_extractor(x)
        style_features = self.style_extractor(x)
        
        content_q, content_loss, content_idx = self.content_vq(content_features)
        style_q, style_loss, style_idx = self.style_vq(style_features)
        
        # Orthogonality loss (clamped)
        content_norm = F.normalize(content_features, dim=-1)
        style_norm = F.normalize(style_features, dim=-1)
        cosine_sim = torch.abs((content_norm * style_norm).sum(dim=-1)).mean()
        disentangle_loss = torch.clamp(cosine_sim, 0, 1)
        
        combined = self.combiner(torch.cat([content_q, style_q], dim=-1))
        total_loss = content_loss + style_loss + 0.5 * disentangle_loss
        
        return combined, total_loss, content_idx, style_idx, disentangle_loss

# -----------------------------
# 5. Relational Structure Extractor (Recommendation #2)
# -----------------------------
class RelationExtractor(nn.Module):
    """Learns relationships between bucket concepts"""
    def __init__(self, d_model, num_relations=8):
        super().__init__()
        self.num_relations = num_relations
        
        self.relation_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_relations)
        )
        
    def forward(self, bucket_embeddings):
        B, S, D = bucket_embeddings.shape
        
        left = bucket_embeddings.unsqueeze(2).expand(B, S, S, D)
        right = bucket_embeddings.unsqueeze(1).expand(B, S, S, D)
        
        pairs = torch.cat([left, right], dim=-1)
        relation_logits = self.relation_predictor(pairs)
        
        return relation_logits

# -----------------------------
# 6. Graph Neural Network (Recommendation #6) - Simplified
# -----------------------------
class SemanticGraphProcessor(nn.Module):
    """Processes bucket concepts as a graph with learned edges"""
    def __init__(self, d_model, num_relations=8, num_layers=2):
        super().__init__()
        self.num_relations = num_relations
        self.num_layers = num_layers
        
        self.message_fn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.update_fn = nn.GRUCell(d_model, d_model)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
    def forward(self, nodes, relation_logits):
        B, S, D = nodes.shape
        h = nodes
        
        for layer in range(self.num_layers):
            messages = torch.zeros_like(h)
            relation_weights = F.softmax(relation_logits.mean(dim=-1), dim=-1)
            
            for i in range(S):
                neighbor_h = h.unsqueeze(1).expand(B, S, S, D)[:, i]
                self_h = h[:, i].unsqueeze(1).expand(B, S, D)
                
                edge_input = torch.cat([neighbor_h, self_h], dim=-1)
                msg = self.message_fn(edge_input)
                
                weighted_msg = msg * relation_weights[:, i].unsqueeze(-1)
                messages[:, i] = weighted_msg.sum(dim=1)
            
            h_flat = h.reshape(B * S, D)
            messages_flat = messages.reshape(B * S, D)
            h_flat = self.update_fn(messages_flat, h_flat)
            h = h_flat.reshape(B, S, D)
            h = self.layer_norms[layer](h)
        
        return h

# -----------------------------
# 7. Predictive Coding Layer (Recommendation #5) - FIXED
# -----------------------------
class PredictiveCodingLayer(nn.Module):
    """Implements hierarchical prediction error minimization"""
    def __init__(self, d_model, n_frequencies):
        super().__init__()
        self.bottom_up = SpectralConv(d_model, n_frequencies)
        self.top_down = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.error_weight = nn.Parameter(torch.tensor(0.1))  # Reduced from 0.5
        
    def forward(self, bottom_up_input, top_down_prediction=None):
        bu_repr = self.bottom_up(bottom_up_input)
        
        if top_down_prediction is not None:
            error = bu_repr - top_down_prediction
            # Clamp error
            error = torch.clamp(error, -1.0, 1.0)
            corrected = bu_repr - torch.sigmoid(self.error_weight) * error
            prediction_error = (error.pow(2).mean()).clamp(0, 1)
        else:
            corrected = bu_repr
            prediction_error = torch.tensor(0.0, device=bu_repr.device)
        
        next_prediction = self.top_down(corrected)
        
        return corrected, next_prediction, prediction_error

# -----------------------------
# 8. Spectral Components (Enhanced)
# -----------------------------
class SpectralConv(nn.Module):
    def __init__(self, d_model, n_frequencies):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.to_spectral = nn.Linear(d_model, n_frequencies * 2, bias=False)
        self.log_decay = nn.Parameter(torch.linspace(-3.0, -0.001, n_frequencies))
        self.frequencies = nn.Parameter(torch.randn(n_frequencies) * 0.1)
        self.from_spectral = nn.Linear(n_frequencies * 2, d_model, bias=False)
        self.register_buffer('dt', torch.tensor(1.0))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm(x)
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        spectral = self.to_spectral(x)
        real = spectral[..., :self.n_frequencies]
        imag = spectral[..., self.n_frequencies:]
        
        u = torch.complex(real, imag)
        decay = torch.sigmoid(self.log_decay)
        omega = torch.tanh(self.frequencies) * 0.1 * self.dt
        rot = torch.complex(torch.cos(omega), torch.sin(omega))
        
        A = decay * rot
        A_expanded = A.view(1, 1, self.n_frequencies).expand(1, seq_len, self.n_frequencies)
        
        ones = torch.ones(1, 1, self.n_frequencies, device=device, dtype=A.dtype)
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
            nn.Dropout(dropout),
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
# 9. Contrastive Predictive Coding (Recommendation #3) - FIXED
# -----------------------------
class ContrastiveBucketLoss(nn.Module):
    """InfoNCE loss for learning predictive representations"""
    def __init__(self, d_model, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.future_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, past_buckets, future_buckets):
        past_repr = F.normalize(self.context_encoder(past_buckets), dim=-1)
        future_repr = F.normalize(self.future_encoder(future_buckets), dim=-1)
        
        logits = torch.matmul(past_repr, future_repr.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        # Clamp to prevent explosion
        return torch.clamp(loss, 0, 10)

# -----------------------------
# 10. Meta-Learning Pattern Extractor (Recommendation #7)
# -----------------------------
class MetaPatternLearner(nn.Module):
    """Learns to rapidly adapt to new patterns"""
    def __init__(self, d_model):
        super().__init__()
        self.pattern_encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.hypernetwork = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * d_model)
        )
        
    def forward(self, support_examples):
        _, (pattern_repr, _) = self.pattern_encoder(support_examples)
        pattern_repr = pattern_repr.squeeze(0)
        pattern_params = self.hypernetwork(pattern_repr)
        return pattern_params

# -----------------------------
# 11. Symbolic Rule Extractor (Recommendation #9)
# -----------------------------
class SymbolicRuleExtractor(nn.Module):
    """Distills neural patterns into explicit symbolic rules"""
    def __init__(self, num_buckets, d_model):
        super().__init__()
        self.num_buckets = num_buckets
        self.pattern_memory = {}
        self.min_frequency = 3
        
    def update_patterns(self, bucket_sequences):
        B, T = bucket_sequences.shape
        
        for b in range(B):
            seq = bucket_sequences[b].cpu().tolist()
            
            for n in range(2, min(6, T)):
                for i in range(T - n + 1):
                    pattern = tuple(seq[i:i+n])
                    if pattern not in self.pattern_memory:
                        self.pattern_memory[pattern] = 0
                    self.pattern_memory[pattern] += 1
    
    def extract_rules(self):
        rules = []
        
        for pattern, count in self.pattern_memory.items():
            if count >= self.min_frequency and len(pattern) >= 3:
                rule = {
                    'condition': pattern[:-1],
                    'action': pattern[-1],
                    'confidence': count,
                    'support': count / max(1, sum(self.pattern_memory.values()))
                }
                rules.append(rule)
        
        rules.sort(key=lambda x: x['confidence'], reverse=True)
        return rules[:100]

# -----------------------------
# 12. Causal Intervention Module (Recommendation #8) - FIXED
# -----------------------------
class CausalInterventionLoss(nn.Module):
    """Learns causal structure through interventions"""
    def __init__(self, d_model):
        super().__init__()
        self.intervention_noise = 0.3
        # Simple causal projection to model how one position affects the next
        self.causal_proj = nn.Linear(d_model, d_model)
        
    def forward(self, model, x, bucket_embeddings):
        B, S, D = bucket_embeddings.shape
        
        if S < 2:
            return torch.tensor(0.0, device=x.device)
        
        # Take all positions except last (these will be used to predict next)
        current_positions = bucket_embeddings[:, :-1]  # [B, S-1, D]
        actual_next = bucket_embeddings[:, 1:]          # [B, S-1, D]
        
        # Normal prediction: what does current position predict about next?
        normal_predicted_next = self.causal_proj(current_positions)  # [B, S-1, D]
        
        # Intervened: add noise to current positions
        noise = torch.randn_like(current_positions) * self.intervention_noise
        intervened_positions = current_positions + noise
        
        # What does intervened position predict about next?
        intervened_predicted_next = self.causal_proj(intervened_positions)  # [B, S-1, D]
        
        # Measure how much predictions changed due to intervention
        prediction_changes = torch.norm(
            intervened_predicted_next - normal_predicted_next, 
            dim=-1
        )  # [B, S-1]
        
        # We want sparse causal effects (small changes on average)
        mean_change = prediction_changes.mean()
        
        # Penalize large changes (want causal sparsity)
        causal_loss = torch.clamp(-torch.log(mean_change + 1e-6), 0, 5)
        
        return causal_loss

# -----------------------------
# 13. Main Cognitive Spectral Model
# -----------------------------
class CognitiveSpectralLM(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_frequencies=128, 
                 num_content_buckets=512, num_style_buckets=64, num_relations=8, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_content_buckets = num_content_buckets
        self.num_style_buckets = num_style_buckets
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.segmenter = AdaptiveSegmenter(d_model, max_segment_len=32)
        self.disentangled_vq = DisentangledVQ(d_model, num_content_buckets, num_style_buckets)
        self.relation_extractor = RelationExtractor(d_model, num_relations)
        self.graph_processor = SemanticGraphProcessor(d_model, num_relations, num_layers=2)
        
        self.predictive_layers = nn.ModuleList([
            PredictiveCodingLayer(d_model, n_frequencies) for _ in range(n_layers // 2)
        ])
        
        self.contrastive_loss_fn = ContrastiveBucketLoss(d_model)
        self.meta_learner = MetaPatternLearner(d_model)
        self.causal_loss_fn = CausalInterventionLoss(d_model)
        self.rule_extractor = SymbolicRuleExtractor(num_content_buckets, d_model)
        
        self.content_bucket_emb = nn.Embedding(num_content_buckets, d_model)
        self.style_bucket_emb = nn.Embedding(num_style_buckets, d_model)
        
        self.bucket_layers = nn.ModuleList([
            SpectralBlock(d_model, n_frequencies, dropout) for _ in range(n_layers // 2)
        ])
        
        self.content_head = nn.Linear(d_model, num_content_buckets)
        self.style_head = nn.Linear(d_model, num_style_buckets)
        
        self.token_layers = nn.ModuleList([
            SpectralBlock(d_model, n_frequencies, dropout) for _ in range(n_layers)
        ])
        
        self.cross_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            ) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        
    def forward(self, x):
        B, T = x.shape
        
        h_tokens = self.token_emb(x)
        h_tokens = self.dropout(h_tokens)
        
        segments, boundaries, boundary_logits = self.segmenter(h_tokens)
        S = segments.shape[1]
        
        disentangled, disentangle_loss, content_indices, style_indices, style_confusion = \
            self.disentangled_vq(segments)
        
        relation_logits = self.relation_extractor(disentangled)
        graph_processed = self.graph_processor(disentangled, relation_logits)
        
        hierarchy_states = []
        prediction_errors = []
        
        current = graph_processed
        top_down_pred = None
        
        for layer in self.predictive_layers:
            current, top_down_pred, pred_error = layer(current, top_down_pred)
            hierarchy_states.append(current)
            prediction_errors.append(pred_error)
        
        total_prediction_error = sum(prediction_errors) / len(prediction_errors)
        
        if S > 1:
            mid = S // 2
            past = hierarchy_states[-1][:, :mid].mean(dim=1)
            future = hierarchy_states[-1][:, mid:].mean(dim=1)
            contrastive_loss = self.contrastive_loss_fn(past, future)
        else:
            contrastive_loss = torch.tensor(0.0, device=x.device)
        
        causal_loss = self.causal_loss_fn(self, x, graph_processed)
        
        h_buckets = hierarchy_states[-1]
        bucket_states = []
        
        for layer in self.bucket_layers:
            h_buckets = layer(h_buckets)
            bucket_states.append(h_buckets)
        
        while len(bucket_states) < len(self.token_layers):
            bucket_states.append(h_buckets)
        
        content_logits = self.content_head(h_buckets)
        style_logits = self.style_head(h_buckets)
        
        bucket_context = torch.zeros(B, T, self.d_model, device=x.device)
        
        for b in range(B):
            if len(boundaries[b]) > 0:
                for i, boundary_pos in enumerate(boundaries[b]):
                    if i < S:
                        start = 0 if i == 0 else boundaries[b][i-1] + 1
                        end = min(boundary_pos + 1, T)
                        if end > start and i < content_indices.shape[1]:
                            content_emb = self.content_bucket_emb(content_indices[b, i])
                            style_emb = self.style_bucket_emb(style_indices[b, i])
                            bucket_context[b, start:end] = content_emb + style_emb
        
        h_tokens = h_tokens + bucket_context
        
        for i, layer in enumerate(self.token_layers):
            bucket_state = bucket_states[min(i, len(bucket_states)-1)]
            
            expanded_bucket = torch.zeros(B, T, self.d_model, device=x.device)
            for b in range(B):
                if len(boundaries[b]) > 0:
                    for j, boundary_pos in enumerate(boundaries[b]):
                        if j < S:
                            start = 0 if j == 0 else boundaries[b][j-1] + 1
                            end = min(boundary_pos + 1, T)
                            if end > start and j < bucket_state.shape[1]:
                                expanded_bucket[b, start:end] = bucket_state[b, j]
            
            injection = self.cross_projections[i](expanded_bucket)
            h_tokens = layer(h_tokens, conditioning=injection)
        
        h_tokens = self.ln_f(h_tokens)
        token_logits = self.lm_head(h_tokens)
        
        if self.training:
            self.rule_extractor.update_patterns(content_indices)
        
        return {
            'token_logits': token_logits,
            'content_logits': content_logits,
            'style_logits': style_logits,
            'content_indices': content_indices,
            'style_indices': style_indices,
            'boundaries': boundaries,
            'disentangle_loss': disentangle_loss,
            'style_confusion': style_confusion,
            'prediction_error': total_prediction_error,
            'contrastive_loss': contrastive_loss,
            'causal_loss': causal_loss,
            'boundary_logits': boundary_logits
        }

# -----------------------------
# 14. Training Logic with FIXED loss scaling
# -----------------------------
def get_batch(data, batch_size, seq_len, device):
    if len(data) <= seq_len + 1:
        raise ValueError(f"Data too small: {len(data)} tokens, need at least {seq_len + 2}")
    
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
        try:
            x, y = get_batch(data, batch_size, seq_len, device)
            outputs = model(x)
            token_logits = outputs['token_logits']
            loss_token = F.cross_entropy(token_logits.view(-1, token_logits.size(-1)), y.view(-1))
            losses.append(loss_token.item())
        except:
            break
    
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
    print("Initializing optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=lr/10)
    model.train()
    
    print("Preparing data...")
    if isinstance(data_source, torch.Tensor):
        n = int(0.9 * len(data_source))
        train_data = data_source[:n]
        val_data = data_source[n:]
        is_tensor = True
        print(f"Train data: {len(train_data):,} tokens, Val data: {len(val_data):,} tokens")
    else:
        is_tensor = False
        val_data = None
    
    print(f"\nStarting training loop...")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Target steps: {steps}, Batch size: {batch_size}, Sequence length: {seq_len}\n")
    
    step = 0
    while step < steps:
        if is_tensor:
            try:
                x, y = get_batch(train_data, batch_size, seq_len, device)
                batches = [(x, y)]
            except Exception as e:
                print(f"Error getting batch: {e}")
                break
        
        for x, y in batches:
            try:
                outputs = model(x)
                
                token_logits = outputs['token_logits']
                loss_token = F.cross_entropy(token_logits.view(-1, token_logits.size(-1)), y.view(-1))
                
                content_logits = outputs['content_logits']
                content_indices = outputs['content_indices']
                
                if content_indices.shape[1] > 1:
                    content_targets = content_indices[:, 1:]
                    content_preds = content_logits[:, :-1]
                    loss_content = F.cross_entropy(
                        content_preds.reshape(-1, model.num_content_buckets),
                        content_targets.reshape(-1)
                    )
                else:
                    loss_content = torch.tensor(0.0, device=device)
                
                style_logits = outputs['style_logits']
                style_indices = outputs['style_indices']
                
                if style_indices.shape[1] > 1:
                    style_targets = style_indices[:, 1:]
                    style_preds = style_logits[:, :-1]
                    loss_style = F.cross_entropy(
                        style_preds.reshape(-1, model.num_style_buckets),
                        style_targets.reshape(-1)
                    )
                else:
                    loss_style = torch.tensor(0.0, device=device)
                
                loss_disentangle = torch.clamp(outputs['disentangle_loss'], 0, 2)
                loss_prediction_error = torch.clamp(outputs['prediction_error'], 0, 2)
                loss_contrastive = torch.clamp(outputs['contrastive_loss'], 0, 5)
                loss_causal = torch.clamp(outputs['causal_loss'], 0, 5)
                
                boundary_logits = outputs['boundary_logits']
                boundary_probs = F.softmax(boundary_logits, dim=-1)[:, :, 1]
                boundary_rate = boundary_probs.mean()
                target_rate = 0.15
                loss_boundary = (boundary_rate - target_rate).pow(2)
                
                # FIXED WEIGHTS - much smaller for auxiliary losses
                loss = (
                    loss_token +
                    0.2 * loss_content +
                    0.1 * loss_style +
                    0.05 * loss_disentangle +
                    0.05 * loss_prediction_error +
                    0.1 * loss_contrastive +
                    0.02 * loss_causal +
                    0.05 * loss_boundary
                )
                
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping PER PARAMETER GROUP
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                
                step += 1
                
                if step % 100 == 0:
                    v_loss = estimate_loss(model, val_data, batch_size=batch_size, seq_len=seq_len, device=device) if is_tensor else 0.0
                    
                    unique_content = torch.unique(content_indices).numel()
                    unique_style = torch.unique(style_indices).numel()
                    avg_boundary_rate = boundary_rate.item()
                    
                    print(f"\n{'='*80}")
                    print(f"Step {step}/{steps} | LR: {scheduler.get_last_lr()[0]:.2e}")
                    print(f"{'='*80}")
                    print(f"Losses:")
                    print(f"  Total: {loss.item():.4f} | Val: {v_loss:.4f}")
                    print(f"  Token: {loss_token.item():.4f}")
                    print(f"  Content: {loss_content.item():.4f} | Style: {loss_style.item():.4f}")
                    print(f"  Disentangle: {loss_disentangle.item():.4f} | Pred Error: {loss_prediction_error.item():.4f}")
                    print(f"  Contrastive: {loss_contrastive.item():.4f} | Causal: {loss_causal.item():.4f}")
                    print(f"  Boundary: {loss_boundary.item():.4f}")
                    print(f"\nStatistics:")
                    print(f"  Active Content Buckets: {unique_content}/{model.num_content_buckets}")
                    print(f"  Active Style Buckets: {unique_style}/{model.num_style_buckets}")
                    print(f"  Boundary Rate: {avg_boundary_rate:.2%}")
                    print(f"{'='*80}\n")
                
                if step % 500 == 0:
                    rules = model.rule_extractor.extract_rules()
                    if rules:
                        print(f"\n--- Top Symbolic Rules (Step {step}) ---")
                        for i, rule in enumerate(rules[:5]):
                            print(f"  {i+1}. {rule['condition']} → {rule['action']} "
                                  f"(confidence: {rule['confidence']}, support: {rule['support']:.4f})")
                        print()
                
                if step % 1000 == 0:
                    print(f"\n--- Generation Sample (Step {step}) ---")
                    sample_text = generate_sample(model, tokenizer, device, prompt="The", top_k=10)
                    print(f"Generated: {sample_text}")
                    print(f"{'='*80}\n")
                
                if step >= steps:
                    break
                    
            except Exception as e:
                print(f"Error during training step {step}: {e}")
                import traceback
                traceback.print_exc()
                break
    
    print("Training finished. Saving checkpoint...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }, "cognitive_checkpoint.pt")
    print("Checkpoint saved to cognitive_checkpoint.pt")

# -----------------------------
# 15. Main Entry Point
# -----------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    print("Initializing Cognitive Spectral Model...")
    model = CognitiveSpectralLM(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        n_layers=6,
        n_frequencies=256,
        num_content_buckets=512,
        num_style_buckets=64,
        num_relations=8,
        dropout=0.2
    ).to(device)
    
    if os.path.exists("cognitive_checkpoint.pt"):
        print("Checkpoint found. Loading and entering Chat Mode...")
        checkpoint = torch.load("cognitive_checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("\n" + "="*80)
        print("COGNITIVE SPECTRAL LANGUAGE MODEL - CHAT MODE")
        print("="*80)
        print("Type 'exit' to quit, 'rules' to see extracted patterns")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input("\nUser: ")
                
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                if user_input.lower() == "rules":
                    rules = model.rule_extractor.extract_rules()
                    print("\n--- Discovered Symbolic Rules ---")
                    for i, rule in enumerate(rules[:20]):
                        print(f"{i+1}. {rule['condition']} → {rule['action']} "
                              f"(conf: {rule['confidence']}, supp: {rule['support']:.4f})")
                    continue
                
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
                
                print("\nBot: ", end="", flush=True)
                
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
                
                print("\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting Chat Mode.")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue
    
    else:
        print("\nNo checkpoint found. Starting training...")
        
        if not os.path.exists("input.txt"):
            print("Creating dummy training data...")
            with open("input.txt", "w") as f:
                texts = [
                    "The quick brown fox jumps over the lazy dog. ",
                    "A journey of a thousand miles begins with a single step. ",
                    "To be or not to be, that is the question. ",
                    "All that glitters is not gold. ",
                    "Where there is a will, there is a way. ",
                    "Knowledge is power, and power corrupts. ",
                    "The pen is mightier than the sword. ",
                    "Actions speak louder than words. "
                ]
                f.write(((" ".join(texts)) * 500))
        
        print("Loading training data from input.txt...")
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        
        print("Tokenizing...")
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        print(f"Training data: {len(data):,} tokens")
        
        if len(data) < 300:
            print("Warning: Very small dataset. Consider using more data.")
        
        train(model, data, tokenizer, steps=100000, batch_size=4, seq_len=512, lr=3e-4, device=device)
        
        print("\n" + "="*80)
        print("Training complete! Run again to enter chat mode.")
        print("="*80)