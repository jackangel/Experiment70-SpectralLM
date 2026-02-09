import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import glob
import time
import signal
import sys
from pathlib import Path
from transformers import GPT2TokenizerFast
import pyarrow.parquet as pq
import requests
import json
import random
from typing import List, Tuple, Dict, Optional
from torch.nn.utils.rnn import pad_sequence

# -----------------------------
# Global variables for cleanup
# -----------------------------
model_global = None
checkpoint_path_global = 'best_model.pt'

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully by saving the model."""
    print("\n\n" + "="*60)
    print("INTERRUPTED - Saving model before exit...")
    print("="*60)
    
    if model_global is not None:
        try:
            torch.save(model_global.state_dict(), checkpoint_path_global)
            print(f"✓ Model saved to {checkpoint_path_global}")
        except Exception as e:
            print(f"✗ Error saving model: {e}")
    
    print("Exiting...")
    print("="*60 + "\n")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# -----------------------------
# Advanced Sampling Functions
# -----------------------------
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    assert logits.dim() == 1  # batch size 1 for now
    top_k = min(top_k, logits.size(-1))  # Safety check
    
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    return logits

def sample_token(logits, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.0, 
                 generated_tokens=None):
    """
    Sample from the model with various techniques:
    - Temperature scaling
    - Top-k filtering
    - Top-p (nucleus) sampling
    - Repetition penalty
    """
    # Apply repetition penalty
    if repetition_penalty != 1.0 and generated_tokens is not None and len(generated_tokens) > 0:
        for token_id in set(generated_tokens[-50:]):  # Only penalize recent tokens
            if token_id < len(logits):
                logits[token_id] /= repetition_penalty
    
    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply top-k and top-p filtering
    filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=top_k, top_p=top_p)
    
    # Check for all -inf (shouldn't happen with proper filtering)
    if torch.isinf(filtered_logits).all():
        filtered_logits = logits  # Fallback to unfiltered
    
    # Sample from the filtered distribution
    probs = F.softmax(filtered_logits, dim=-1)
    
    # Handle potential NaN in probabilities
    if torch.isnan(probs).any():
        return torch.argmax(logits).item()
    
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token

# -----------------------------
# Helper: Masked Batch Preparation
# -----------------------------
def prepare_masked_batch(tokenizer, examples, device, max_len=512):
    """
    Prepares a batch where User input is masked (label = -100)
    so the model only learns to generate the Assistant response.
    """
    PROMPT_START = "### User:\n"
    RESPONSE_START = "\n### Assistant:\n"
    
    input_ids_list = []
    labels_list = []
    
    for ex in examples:
        context = ex.get('context', '')
        
        if 'User:' in context:
            last_user_idx = context.rfind("User:")
            clean_query = context[last_user_idx:].replace("User:", "").replace("###", "").strip()
        else:
            clean_query = context.strip()
            
        expert_response = ex['expert_response'].strip()
        
        prompt_text = f"{PROMPT_START}{clean_query}{RESPONSE_START}"
        prompt_ids = tokenizer.encode(prompt_text)
        
        response_ids = tokenizer.encode(expert_response + tokenizer.eos_token)
        
        full_ids = prompt_ids + response_ids
        full_labels = [-100] * len(prompt_ids) + response_ids
        
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
            full_labels = full_labels[:max_len]
            
        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        labels_list.append(torch.tensor(full_labels, dtype=torch.long))
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    batch_inputs = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    batch_labels = pad_sequence(labels_list, batch_first=True, padding_value=-100).to(device)
    
    return batch_inputs, batch_labels

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
        for i, file in enumerate(self.parquet_files[:5]):
            print(f"  {i+1}. {Path(file).name}")
    
    def load_file_in_chunks(self, file_path):
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        
        print(f"\nLoading: {Path(file_path).name}")
        all_tokens = []
        
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            df = batch.to_pandas()
            if self.text_column not in df.columns:
                raise ValueError(f"Column '{self.text_column}' not found.")
            
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
# Cross-Attention Module
# -----------------------------
class CrossAttention(nn.Module):
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
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key_value):
        batch_size, seq_len, _ = query.shape
        q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
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
        
        self.to_spectral = nn.Linear(d_model, n_frequencies * 2, bias=False)
        self.log_decay = nn.Parameter(torch.zeros(n_frequencies))
        self.frequencies = nn.Parameter(torch.randn(n_frequencies) * 0.01)
        self.from_spectral = nn.Linear(n_frequencies * 2, d_model, bias=False)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        if use_cross_attn:
            self.cross_attn = CrossAttention(d_model, n_heads)
            self.norm_cross = nn.LayerNorm(d_model)
        
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
                if layer.bias is not None: nn.init.zeros_(layer.bias)
    
    def forward(self, x, prev_layer_output=None):
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        residual = x
        x = self.norm1(x)
        
        spectral = self.to_spectral(x)
        real = spectral[..., :self.n_frequencies]
        imag = spectral[..., self.n_frequencies:]
        
        decay = torch.sigmoid(self.log_decay)
        omega = torch.tanh(self.frequencies) * 0.1
        cos_omega = torch.cos(omega * self.dt)
        sin_omega = torch.sin(omega * self.dt)
        
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
        
        if self.use_cross_attn and prev_layer_output is not None:
            residual2 = x
            x = self.norm_cross(x)
            x = self.cross_attn(x, prev_layer_output)
            x = residual2 + x * 0.1
        
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
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            SpectralLayerWithAttention(
                d_model, n_frequencies, n_heads, use_cross_attn=(i > 0)
            ) for i in range(n_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        for layer in self.layers:
            layer._init_weights()
    
    def forward(self, sequence):
        seq_len = sequence.shape[1]
        x = self.token_embedding(sequence)
        x = x + self.pos_embedding[:, :seq_len, :]
        prev_output = None
        for layer in self.layers:
            x = layer(x, prev_output)
            prev_output = x
        x = self.output_norm(x)
        logits = self.output_head(x)
        return logits

# -----------------------------
# LM Studio API Interface
# -----------------------------
class LMStudioClient:
    def __init__(self, base_url="http://192.168.1.91:1234"):
        self.base_url = base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        
    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.8, 
                         top_p: float = 0.9, top_k: int = 50) -> str:
        """Updated to include sampling parameters"""
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False
        }
        try:
            response = requests.post(
                self.endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=60
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error communicating with LM Studio: {e}")
            return ""
    
    def rate_response(self, query: str, response: str) -> Tuple[float, str]:
        rating_prompt = f"""Rate this AI response to: "{query}"\nResponse: "{response}"\nRate 0-10 (number only)."""
        try:
            rating_text = self.generate_response(rating_prompt, max_tokens=10, temperature=0.1)
            import re
            numbers = re.findall(r'\d+\.?\d*', rating_text)
            if numbers:
                return float(numbers[0]) / 10.0, rating_text
            return 0.5, "No rating"
        except:
            return 0.5, "Error"
    
    def generate_conversation_topic(self) -> str:
        """Generate a random conversation topic"""
        system_instruction = (
            "You are a casual human user starting a chat with an AI. "
            "Output only one short sentence to start a conversation. "
            "Do not include quotes. Do not say 'Here is a topic'. "
            "Examples: 'What is the capital of France?', 'Tell me a joke.', 'I'm feeling sad today.'"
        )
        prompt = f"{system_instruction}\n\nGenerate a new conversation starter now:"

        try:
            topic = self.generate_response(prompt, max_tokens=60, temperature=1.1, top_p=0.95)
            topic = topic.replace("Here is a conversation starter:", "").replace("Sure!", "").replace('"', '').strip()
            return topic if topic else "Hello, who are you?"
        except Exception as e:
            print(f"[WARNING] Failed to generate topic: {e}")
            return "Hello, who are you?"

# -----------------------------
# Conversation Manager (UPDATED WITH SAMPLING)
# -----------------------------
class ConversationManager:
    def __init__(self, your_model, tokenizer, lm_studio_client: LMStudioClient, device='cpu'):
        self.your_model = your_model
        self.tokenizer = tokenizer
        self.lm_studio = lm_studio_client
        self.device = device
        
    def generate_from_your_model(self, conversation_history: List[Dict], max_tokens: int = 300,
                                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
                                 repetition_penalty: float = 1.1) -> str:
        """Updated generation with proper sampling parameters"""
        self.your_model.eval()
        prompt = self._format_conversation(conversation_history)
        
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt)
            if len(input_ids) > 1024 - max_tokens:
                input_ids = input_ids[-(1024 - max_tokens):]
            
            generated = input_ids.copy()
            for _ in range(max_tokens):
                context = torch.tensor([generated[-256:]], dtype=torch.long, device=self.device)
                logits = self.your_model(context)
                
                # Get logits for the last token
                last_logits = logits[0, -1]
                
                # Check for NaN
                if torch.isnan(last_logits).any():
                    break
                
                # Use advanced sampling
                next_token = sample_token(
                    last_logits, 
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_tokens=generated
                )
                
                generated.append(next_token)
                if next_token == self.tokenizer.eos_token_id:
                    break
            
            full_text = self.tokenizer.decode(generated)
            response = full_text[len(self.tokenizer.decode(input_ids)):]
        
        self.your_model.train()
        return response.strip()
    
    def _format_conversation(self, history: List[Dict]) -> str:
        """Format history, keeping ONLY the last few turns"""
        PROMPT_START = "### User:\n"
        RESPONSE_START = "\n### Assistant:\n"
        
        recent_history = history[-3:]
        
        formatted = "The following is a conversation between a User and an AI Assistant.\n\n"
        
        for turn in recent_history:
            role = turn['role']
            content = turn['content'].strip()
            if len(content) > 600:
                content = content[:600] + "..."
            
            if role == 'user':
                formatted += f"{PROMPT_START}{content}"
            elif role == 'assistant':
                formatted += f"{RESPONSE_START}{content}\n"
        
        if history and history[-1]['role'] == 'user':
            formatted += f"{RESPONSE_START}"
            
        return formatted
    
    def have_conversation(self, starter_prompt: str, num_turns: int = 5, your_model_starts: bool = False) -> Dict:
        """Conducts a training conversation using Teacher-Student forcing"""
        conversation_history = []
        training_examples = []
        
        print(f"\n{'='*80}")
        print(f"TOPIC: {starter_prompt}")
        print(f"{'='*80}\n")
        
        current_user_input = starter_prompt
        
        for turn_num in range(num_turns):
            print(f"\n--- Turn {turn_num + 1}/{num_turns} ---\n")
            
            conversation_history.append({'role': 'user', 'content': current_user_input})
            print(f"👤 User (Simulated): {current_user_input}")
            
            # Student attempt with proper sampling
            your_response = self.generate_from_your_model(
                conversation_history,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1
            )
            print(f"🤖 Your Model (Student): {your_response[:200]}..." if len(your_response) > 200 else f"🤖 Your Model (Student): {your_response}")
            
            # Teacher correction
            teacher_prompt = (
                f"SYSTEM: You are an expert AI Assistant. "
                f"The user just said: '{current_user_input}'.\n"
                f"TASK: Provide a perfect, helpful response. "
                f"Keep it concise (2-3 sentences max). Do not ramble."
            )
            print("🧠 LM Studio (Teacher) generating ideal response...")
            expert_response = self.lm_studio.generate_response(
                teacher_prompt, 
                max_tokens=300, 
                temperature=0.7,
                top_p=0.9,
                top_k=50
            )
            
            if not expert_response:
                expert_response = "I understand. Please tell me more."
            print(f"✅ Teacher: {expert_response}")
            
            training_examples.append({
                'context': self._format_conversation(conversation_history[:-1]), 
                'expert_response': expert_response,
                'your_response': your_response
            })
            
            conversation_history.append({'role': 'assistant', 'content': expert_response})
            
            if turn_num < num_turns - 1:
                prev_turn = f"AI Assistant said: \"{expert_response}\""
                
                sim_prompt = (
                    f"OBJECTIVE: Roleplay as a Human User chatting with an AI.\n"
                    f"CONTEXT: {prev_turn}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Respond to the AI naturally.\n"
                    f"2. Be curious, confused, or ask a follow-up question.\n"
                    f"3. CRITICAL: Do NOT act like an AI. Do NOT ask 'How can I help you?'. Do NOT offer assistance.\n"
                    f"4. You are the one SEEKING help or conversation.\n"
                    f"5. Keep it short (1 sentence).\n\n"
                    f"Human User Response:"
                )
                
                next_input = self.lm_studio.generate_response(
                    sim_prompt, 
                    max_tokens=60, 
                    temperature=0.9,
                    top_p=0.95
                )
                
                cleaned_input = next_input.replace("User:", "").replace("Human:", "").replace("AI:", "").strip()
                
                if "how can i help" in cleaned_input.lower() or "is there anything else" in cleaned_input.lower():
                    print("⚠️ Simulator broke character. Forcing fallback.")
                    cleaned_input = "That's interesting. Tell me more."
                
                current_user_input = cleaned_input
        
        return {
            'conversation_history': conversation_history,
            'training_examples': training_examples,
            'topic': starter_prompt
        }

# -----------------------------
# Training Functions
# -----------------------------
def conversational_instruction_tune(
    model, tokenizer, lm_studio_client, device,
    num_rounds=None, conversations_per_round=3, turns_per_conversation=5,
    training_batch_size=4, learning_rate=1e-5, save_path='conversational_tuned_model.pt',
    continuous=False, max_seq_len=512, stride=None
):
    global checkpoint_path_global
    checkpoint_path_global = save_path
    
    print("\n" + "="*80)
    print("CONVERSATIONAL INSTRUCTION TUNING (MASKED & TEACHER FORCING)")
    print("="*80)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    conversation_manager = ConversationManager(model, tokenizer, lm_studio_client, device)
    all_training_examples = []
    round_num = 0
    
    try:
        while True:
            round_num += 1
            if not continuous and num_rounds is not None and round_num > num_rounds:
                break
            
            print(f"\nROUND {round_num} | Collecting Data...")
            round_examples = []
            
            for _ in range(conversations_per_round):
                if random.random() < 0.5:
                    topic = lm_studio_client.generate_conversation_topic()
                else:
                    topic = random.choice([
                        "Hello", "Who are you?", "Are you a human?", "Good morning",
                        "What can you do?", "Explain gravity", "Why is the sky blue?",
                        "What is photosynthesis?", "What is the capital of France?",
                        "How do cars work?", "Tell me a joke", "Write a short poem",
                        "Tell me a story about a robot", "I'm bored, entertain me",
                        "Give me a fun fact", "Help me", "I'm feeling sad today",
                        "How do I focus better?", "Give me a cooking tip", "I need motivation",
                        "What is 25 plus 25?", "Write a Python hello world script",
                        "Compare cats and dogs", "What is the internet?", "Solve a riddle for me"
                    ])
                
                result = conversation_manager.have_conversation(topic, turns_per_conversation)
                round_examples.extend(result['training_examples'])
                time.sleep(1)
            
            all_training_examples.extend(round_examples)
            
            if not round_examples:
                continue
            
            print(f"\n🎓 Training on {len(round_examples)} new interaction examples...")
            training_pool = all_training_examples[-100:]
            model.train()
            
            for epoch in range(5):
                random.shuffle(training_pool)
                epoch_loss = 0
                steps = 0
                
                for i in range(0, len(training_pool), training_batch_size):
                    batch = training_pool[i : i + training_batch_size]
                    if not batch:
                        continue
                    
                    input_ids, labels = prepare_masked_batch(tokenizer, batch, device, max_len=max_seq_len)
                    
                    logits = model(input_ids)
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss = F.cross_entropy(
                        shift_logits.view(-1, model.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    steps += 1
                
                print(f"  Epoch {epoch+1} Loss: {epoch_loss/steps:.4f}" if steps > 0 else "  Epoch skipped")
            
            torch.save(model.state_dict(), save_path)
            
            print("\n🧪 Quick Test:")
            test_prompt = "Hello! How are you?"
            model.eval()
            formatted = f"### User:\n{test_prompt}\n### Assistant:\n"
            inp = torch.tensor([tokenizer.encode(formatted)], device=device)
            
            with torch.no_grad():
                gen = inp
                for _ in range(50):
                    ctx = torch.tensor([gen[0, -256:].tolist()], device=device)
                    logits = model(ctx)
                    
                    # Use advanced sampling for test
                    last_logits = logits[0, -1]
                    next_token = sample_token(
                        last_logits,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        generated_tokens=gen[0].tolist()
                    )
                    
                    gen = torch.cat([gen, torch.tensor([[next_token]], device=device)], dim=1)
                    if next_token == tokenizer.eos_token_id:
                        break
            
            out = tokenizer.decode(gen[0]).replace(formatted, "")
            print(f"Q: {test_prompt}\nA: {out.strip()}\n")
            
    except KeyboardInterrupt:
        print("Training stopped. Saving model.")
        torch.save(model.state_dict(), save_path)

# -----------------------------
# Standard Instruction Tune (Kernel)
# -----------------------------
def instruction_tune(model, tokenizer, lm_studio_client, device, num_rounds=50, examples_per_round=5,
                     training_batch_size=4, learning_rate=1e-5, save_path='instruction_tuned_model.pt'):
    print("Starting Kernel-Based Instruction Tuning...")
    conversational_instruction_tune(
        model, tokenizer, lm_studio_client, device, 
        num_rounds=num_rounds, conversations_per_round=examples_per_round, 
        turns_per_conversation=1,
        save_path=save_path, continuous=False
    )

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

def sample_text(model, start_token, max_len=200, temperature=0.8, top_k=50, top_p=0.9, 
                repetition_penalty=1.1, device='cpu'):
    """Updated sampling function with proper parameters"""
    model.eval()
    generated = [start_token]
    
    with torch.no_grad():
        for _ in range(max_len):
            context = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
            logits = model(context)
            last_logits = logits[0, -1]
            
            if torch.isnan(last_logits).any():
                break
            
            next_token = sample_token(
                last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated
            )
            
            generated.append(next_token)
    
    model.train()
    return generated

def chat_mode(model, device, tokenizer):
    """Updated chat mode with proper sampling"""
    model.eval()
    print("\n" + "="*60 + "\nCHAT MODE (Type 'quit' to stop)\n" + "="*60)
    print("Format: ### User: [Your input] -> ### Assistant: [Response]")
    
    PROMPT_START = "### User:\n"
    RESPONSE_START = "\n### Assistant:\n"
    
    # Configurable sampling parameters
    temperature = 0.8
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.1
    
    print(f"\nSampling Settings: temp={temperature}, top_k={top_k}, top_p={top_p}, rep_penalty={repetition_penalty}")
    print("Type 'config' to adjust settings\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.lower() == 'config':
                print("\nCurrent settings:")
                print(f"  Temperature: {temperature}")
                print(f"  Top-k: {top_k}")
                print(f"  Top-p: {top_p}")
                print(f"  Repetition penalty: {repetition_penalty}")
                
                try:
                    new_temp = input(f"New temperature ({temperature}): ").strip()
                    if new_temp: temperature = float(new_temp)
                    
                    new_k = input(f"New top-k ({top_k}): ").strip()
                    if new_k: top_k = int(new_k)
                    
                    new_p = input(f"New top-p ({top_p}): ").strip()
                    if new_p: top_p = float(new_p)
                    
                    new_rep = input(f"New repetition penalty ({repetition_penalty}): ").strip()
                    if new_rep: repetition_penalty = float(new_rep)
                    
                    print("Settings updated!")
                except:
                    print("Invalid input, keeping current settings")
                continue
            
            if not user_input:
                continue
            
            formatted_input = f"{PROMPT_START}{user_input}{RESPONSE_START}"
            input_ids = tokenizer.encode(formatted_input)
            
            generated = input_ids.copy()
            with torch.no_grad():
                for i in range(200):
                    context = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
                    logits = model(context)
                    last_logits = logits[0, -1]
                    
                    if torch.isnan(last_logits).any():
                        break
                    
                    next_token = sample_token(
                        last_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        generated_tokens=generated
                    )
                    
                    generated.append(next_token)
                    if next_token == tokenizer.eos_token_id:
                        break
            
            full_text = tokenizer.decode(generated)
            response = full_text.replace(formatted_input, "").strip()
            print(f"Model: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    vocab_size = tokenizer.vocab_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    d_model = 384
    n_frequencies = 64
    n_layers = 4
    n_heads = 4
    batch_size = 4
    seq_len = 128
    
    model = HybridSpectralLM(vocab_size, d_model, n_frequencies, n_layers, n_heads).to(device)
    model_global = model
    
    checkpoint_path = 'best_model.pt'
    checkpoint_path_global = checkpoint_path
    best_loss = float('inf')
    
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint: {checkpoint_path}")
        print("1. Continue Pre-training (Parquet)")
        print("2. Chat Mode")
        print("3. Conversational Instruction Tune (Recommended)")
        
        choice = input("Choice (1/2/3): ").strip()
        
        if choice == '1':
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            folder_path = input("Parquet folder: ").strip()
            text_col = input("Text column: ").strip()
            data_loader = ParquetDataLoader(folder_path, text_col, tokenizer)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            # train_on_parquet_files would need to be implemented
            print("Parquet training not implemented in this update")
            
        elif choice == '2':
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            chat_mode(model, device, tokenizer)
            
        elif choice == '3':
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            url = input("LM Studio URL (default http://192.168.1.91:1234): ").strip() or "http://192.168.1.91:1234"
            client = LMStudioClient(url)
            conversational_instruction_tune(
                model, tokenizer, client, device,
                save_path='conversational_tuned_model.pt',
                continuous=True
            )
            
    else:
        print("No checkpoint found. Please run pre-training first.")