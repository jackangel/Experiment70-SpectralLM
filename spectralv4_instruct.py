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
from typing import List, Tuple, Dict

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
# Parquet Data Loader
# -----------------------------
class ParquetDataLoader:
    def __init__(self, folder_path, text_column, tokenizer, chunk_size=10000):
        """
        Load parquet files from a folder recursively.
        
        Args:
            folder_path: Path to folder containing parquet files
            text_column: Name of the column containing text data
            tokenizer: Tokenizer to use for encoding
            chunk_size: Number of rows to process at a time to avoid OOM
        """
        self.folder_path = Path(folder_path)
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        
        # Find all parquet files recursively
        self.parquet_files = sorted(glob.glob(str(self.folder_path / "**" / "*.parquet"), recursive=True))
        
        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {folder_path}")
        
        print(f"Found {len(self.parquet_files)} parquet files")
        for i, file in enumerate(self.parquet_files[:5]):
            print(f"  {i+1}. {Path(file).name}")
        if len(self.parquet_files) > 5:
            print(f"  ... and {len(self.parquet_files) - 5} more files")
    
    def load_file_in_chunks(self, file_path):
        """
        Generator that yields tokenized data in chunks to avoid OOM.
        """
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        
        print(f"\nLoading: {Path(file_path).name}")
        print(f"Total rows: {total_rows:,}")
        
        all_tokens = []
        
        # Read in batches
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            df = batch.to_pandas()
            
            if self.text_column not in df.columns:
                raise ValueError(f"Column '{self.text_column}' not found in {file_path}. Available columns: {list(df.columns)}")
            
            # Process each text entry
            for text in df[self.text_column]:
                if isinstance(text, str) and text.strip():
                    tokens = self.tokenizer.encode(text)
                    all_tokens.extend(tokens)
            
            # Yield when we have enough tokens
            if len(all_tokens) > 1000000:  # Yield every 1M tokens
                yield torch.tensor(all_tokens, dtype=torch.long)
                all_tokens = []
        
        # Yield remaining tokens
        if all_tokens:
            yield torch.tensor(all_tokens, dtype=torch.long)
    
    def get_file_count(self):
        """Return the number of parquet files (epochs)."""
        return len(self.parquet_files)
    
    def iterate_files(self):
        """Iterate through all parquet files."""
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
# LM Studio API Interface
# -----------------------------
class LMStudioClient:
    def __init__(self, base_url="http://192.168.1.91:1234"):
        """
        Interface to LM Studio running on another machine.
        
        Args:
            base_url: Base URL for LM Studio API (default: http://192.168.1.91:1234)
        """
        self.base_url = base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        
    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.8) -> str:
        """
        Generate a response from the LM Studio model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with LM Studio: {e}")
            return ""
    
    def rate_response(self, query: str, response: str) -> Tuple[float, str]:
        rating_prompt = f"""You are a response quality evaluator. Rate the following response.

QUERY: {query}

RESPONSE: {response}

Rate from 0-10 where:
- 0-3: Poor (unhelpful, irrelevant, or inappropriate)
- 4-6: Adequate (addresses query but lacks quality)
- 7-8: Good (helpful, clear, appropriate)
- 9-10: Excellent (exceptionally helpful and well-crafted)

YOUR RATING (number only, first line): """

        try:
            rating_text = self.generate_response(rating_prompt, max_tokens=150, temperature=0.1)
            
            # Print for debugging
            print(f"\n[DEBUG] Rating response:\n{rating_text}\n")
            
            # Parse first number found
            import re
            numbers = re.findall(r'\d+\.?\d*', rating_text)
            
            if numbers:
                score = float(numbers[0])
                score = max(0.0, min(10.0, score))
                explanation = rating_text
                
                print(f"[DEBUG] Parsed score: {score}/10 = {score/10.0}")
                
                return score / 10.0, explanation
            else:
                print(f"[WARNING] No number found in: {rating_text}")
                return 0.5, "Could not parse rating"
                
        except Exception as e:
            print(f"[ERROR] Rating failed: {e}")
            return 0.5, f"Rating error: {e}"
    
    def generate_conversation_topic(self) -> str:
        """
        Generate a random conversation topic using the LM Studio model.
        
        Returns:
            A conversation topic/starter prompt
        """
        topic_prompt = """Generate a single interesting conversation topic or discussion prompt. 
Make it thought-provoking, open-ended, and suitable for a meaningful dialogue.
It could be about philosophy, science, creativity, emotions, practical advice, or any engaging subject.

Just provide the topic or question directly, nothing else."""

        try:
            topic = self.generate_response(topic_prompt, max_tokens=100, temperature=1.0)
            return topic.strip() if topic else random.choice(CONVERSATION_STARTERS)
        except Exception as e:
            print(f"[WARNING] Failed to generate topic: {e}. Using preset.")
            return random.choice(CONVERSATION_STARTERS)

# -----------------------------
# Discussion Kernels
# -----------------------------
DISCUSSION_KERNELS = [
    # Questions and requests
    "What is the weather like today",
    "Can you help me understand quantum physics",
    "Tell me a story about a brave knight",
    "How do I bake chocolate chip cookies",
    "Explain photosynthesis to me",
    "What's your favorite color",
    "Help me write an email to my boss",
    "I'm feeling sad today",
    "Can you recommend a good book",
    "What should I do this weekend",
    
    # Statements that might need responses
    "I just got a new puppy",
    "The sky is so beautiful tonight",
    "I'm learning to play guitar",
    "My computer keeps crashing",
    "I love reading science fiction",
    
    # Conversational
    "Good morning",
    "How are you",
    "That's interesting",
    "I don't understand",
    "Thanks for your help",
    
    # Complex queries
    "I'm trying to decide between two job offers",
    "My friend is upset with me",
    "I want to learn programming but don't know where to start",
    "Can you explain the difference between AI and machine learning",
    "I'm planning a trip to Japan",
    
    # Emotional/supportive
    "I'm worried about my exam tomorrow",
    "I feel lonely sometimes",
    "I'm proud of myself for finishing this project",
    "I miss my family",
    
    # Creative
    "Write a poem about the ocean",
    "Give me an idea for a science fiction story",
    "What would happen if gravity suddenly stopped",
    "Describe a perfect day",
    
    # Practical
    "My plant is dying what should I do",
    "How can I be more productive",
    "What's a good way to exercise at home",
    "I need advice on budgeting",
]

# -----------------------------
# Conversation Topics
# -----------------------------
CONVERSATION_STARTERS = [
    # Open-ended discussions
    "Let's talk about creativity. What makes something creative?",
    "I'm curious about your thoughts on consciousness. What is it?",
    "Can you explain the concept of time to me? I find it confusing.",
    "What do you think makes a good friend?",
    "Tell me about learning. How do you learn new things?",
    
    # Problem-solving dialogues
    "I'm trying to understand quantum mechanics. Can you help me?",
    "I want to become a better writer. Where should I start?",
    "How would you approach solving climate change?",
    "I'm learning programming and feeling overwhelmed. Any advice?",
    
    # Philosophical discussions
    "What is the meaning of life, in your view?",
    "Is artificial intelligence truly intelligent?",
    "What's more important: being right or being kind?",
    "Do you think free will exists?",
    
    # Storytelling
    "Let's create a story together about a character who discovers a hidden world.",
    "Tell me about an imaginary civilization that lives underwater.",
    "What if humans could communicate with animals? Walk me through that world.",
    
    # Scientific explanations
    "Explain photosynthesis like I'm five years old.",
    "How do black holes work? I'm really curious.",
    "What is DNA and why is it important?",
    "Can you explain evolution to me?",
    
    # Emotional/empathetic
    "I've been feeling anxious lately. Let's talk about anxiety.",
    "What makes people happy? I want to understand happiness better.",
    "How do you deal with failure?",
    "Tell me about hope. What does it mean to have hope?",
    
    # Creative scenarios
    "If you could redesign education from scratch, what would you do?",
    "Imagine a world without money. How would it work?",
    "What would you ask an alien civilization if you met them?",
    
    # Practical knowledge
    "I want to start exercising but don't know how. Guide me.",
    "How can someone be more productive without burning out?",
    "What's the best way to learn a new language?",
    "How do you build good habits?",
    
    # Deep conversations
    "What is beauty? How do we recognize it?",
    "Let's discuss ethics. What makes an action right or wrong?",
    "What role does suffering play in human life?",
    "How do we know what's real versus what's illusion?",
    
    # Teaching moments
    "Can you teach me about economics? Start with the basics.",
    "I don't understand how computers work. Explain it to me.",
    "What is music theory? Help me understand it.",
    "How does the internet actually work?",
]

# -----------------------------
# Multi-Turn Conversation Manager
# -----------------------------
class ConversationManager:
    def __init__(self, your_model, tokenizer, lm_studio_client: LMStudioClient, device='cpu'):
        self.your_model = your_model
        self.tokenizer = tokenizer
        self.lm_studio = lm_studio_client
        self.device = device
        
    def generate_from_your_model(self, conversation_history: List[Dict], max_tokens: int = 300) -> str:
        """Generate response from your model given conversation history."""
        self.your_model.eval()
        
        # Format conversation history into a prompt
        prompt = self._format_conversation(conversation_history)
        
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt)
            generated = input_ids.copy()
            
            for _ in range(max_tokens):
                context = torch.tensor([generated[-256:]], dtype=torch.long, device=self.device)
                logits = self.your_model(context)
                logits = logits[0, -1] / 0.8
                
                if torch.isnan(logits).any():
                    break
                
                probs = F.softmax(logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                
                if next_token == self.tokenizer.eos_token_id:
                    break
            
            full_text = self.tokenizer.decode(generated)
            response = full_text[len(prompt):].strip()
        
        self.your_model.train()
        return response
    
    def _format_conversation(self, history: List[Dict]) -> str:
        """Format conversation history into a prompt."""
        formatted = ""
        for turn in history:
            role = turn['role']
            content = turn['content']
            if role == 'user':
                formatted += f"User: {content}\n"
            elif role == 'assistant':
                formatted += f"Assistant: {content}\n"
        formatted += "Assistant: "
        return formatted
    
    def have_conversation(self, starter_prompt: str, num_turns: int = 5, 
                         your_model_starts: bool = False) -> Dict:
        """
        Have a multi-turn conversation between your model and LM Studio.
        
        Args:
            starter_prompt: Initial topic/question to discuss
            num_turns: How many back-and-forth exchanges
            your_model_starts: If True, your model responds first
            
        Returns:
            Dictionary with conversation data and training examples
        """
        conversation_history = []
        training_examples = []
        
        print(f"\n{'='*80}")
        print(f"CONVERSATION TOPIC: {starter_prompt}")
        print(f"{'='*80}\n")
        
        # Add the starter prompt
        conversation_history.append({
            'role': 'user',
            'content': starter_prompt
        })
        
        for turn_num in range(num_turns):
            print(f"\n--- Turn {turn_num + 1}/{num_turns} ---\n")
            
            # Determine who speaks
            if (turn_num == 0 and your_model_starts) or (turn_num > 0 and turn_num % 2 == 1):
                # Your model's turn
                print("🤖 Your Model is thinking...")
                your_response = self.generate_from_your_model(conversation_history)
                
                conversation_history.append({
                    'role': 'assistant',
                    'content': your_response
                })
                
                print(f"Your Model: {your_response}\n")
                
                # Get expert's response to continue the conversation
                expert_prompt = self._format_conversation(conversation_history)
                print("🧠 LM Studio is responding...")
                expert_response = self.lm_studio.generate_response(
                    expert_prompt, 
                    max_tokens=500,
                    temperature=0.9
                )
                
                if not expert_response:
                    print("LM Studio failed to respond, ending conversation.")
                    break
                
                conversation_history.append({
                    'role': 'user',
                    'content': expert_response
                })
                
                print(f"LM Studio: {expert_response}\n")
                
                # Rate your model's response
                rating, explanation = self.lm_studio.rate_response(
                    conversation_history[-3]['content'] if len(conversation_history) >= 3 else starter_prompt,
                    your_response
                )
                
                print(f"📊 Rating: {rating:.2f}/1.0 - {explanation}\n")
                
                # Create training example
                training_examples.append({
                    'context': self._format_conversation(conversation_history[:-2]),
                    'your_response': your_response,
                    'expert_response': expert_response,
                    'rating': rating,
                    'loss_weight': 1.0 - rating + 0.1,
                    'explanation': explanation,
                    'turn': turn_num
                })
                
            else:
                # LM Studio's turn
                print("🧠 LM Studio is thinking...")
                expert_prompt = self._format_conversation(conversation_history)
                expert_response = self.lm_studio.generate_response(
                    expert_prompt,
                    max_tokens=500,
                    temperature=0.9
                )
                
                if not expert_response:
                    print("LM Studio failed to respond, ending conversation.")
                    break
                
                conversation_history.append({
                    'role': 'assistant',
                    'content': expert_response
                })
                
                print(f"LM Studio: {expert_response}\n")
                
                # Your model responds
                print("🤖 Your Model is responding...")
                your_response = self.generate_from_your_model(conversation_history)
                
                conversation_history.append({
                    'role': 'user',
                    'content': your_response
                })
                
                print(f"Your Model: {your_response}\n")
                
                # Rate your model's response
                rating, explanation = self.lm_studio.rate_response(
                    conversation_history[-3]['content'] if len(conversation_history) >= 3 else starter_prompt,
                    your_response
                )
                
                print(f"📊 Rating: {rating:.2f}/1.0 - {explanation}\n")
                
                # Create training example
                training_examples.append({
                    'context': self._format_conversation(conversation_history[:-2]),
                    'your_response': your_response,
                    'expert_response': expert_response,
                    'rating': rating,
                    'loss_weight': 1.0 - rating + 0.1,
                    'explanation': explanation,
                    'turn': turn_num
                })
            
            time.sleep(1)  # Rate limiting
        
        print(f"\n{'='*80}")
        print(f"CONVERSATION COMPLETE")
        print(f"Total turns: {len(training_examples)}")
        avg_rating = sum(ex['rating'] for ex in training_examples) / len(training_examples) if training_examples else 0
        print(f"Average rating: {avg_rating:.2f}/1.0")
        print(f"{'='*80}\n")
        
        return {
            'conversation_history': conversation_history,
            'training_examples': training_examples,
            'avg_rating': avg_rating,
            'topic': starter_prompt
        }

# -----------------------------
# Instruction Tuning Dataset Generator
# -----------------------------
class ConversationalDataGenerator:
    def __init__(self, model, tokenizer, lm_studio_client: LMStudioClient, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.lm_studio = lm_studio_client
        self.device = device
        
    def generate_training_example(self, kernel: str) -> Dict:
        """
        Generate a training example through conversation with LM Studio.
        
        Returns:
            Dictionary with query, your_model_response, expert_response, rating, loss_weight
        """
        # 1. Get response from your model
        your_response = self._generate_from_your_model(kernel)
        
        # 2. Get response from expert (LM Studio)
        expert_response = self.lm_studio.generate_response(kernel)
        
        if not expert_response:
            print("Failed to get expert response, skipping...")
            return None
        
        # 3. Get quality rating of your model's response
        rating, explanation = self.lm_studio.rate_response(kernel, your_response)
        
        print(f"\n{'='*60}")
        print(f"Query: {kernel}")
        print(f"\nYour Model: {your_response[:200]}...")
        print(f"\nExpert: {expert_response[:200]}...")
        print(f"\nRating: {rating:.2f}/1.0")
        print(f"Explanation: {explanation}")
        print(f"{'='*60}\n")
        
        # 4. Create training example
        loss_weight = 1.0 - rating + 0.1
        
        return {
            'query': kernel,
            'your_response': your_response,
            'expert_response': expert_response,
            'rating': rating,
            'loss_weight': loss_weight,
            'explanation': explanation
        }
    
    def _generate_from_your_model(self, prompt: str, max_tokens: int = 300) -> str:
        """Generate response from your model."""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt)
            generated = input_ids.copy()
            
            for _ in range(max_tokens):
                context = torch.tensor([generated[-256:]], dtype=torch.long, device=self.device)
                logits = self.model(context)
                logits = logits[0, -1] / 0.8
                
                if torch.isnan(logits).any():
                    break
                
                probs = F.softmax(logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                
                if next_token == self.tokenizer.eos_token_id:
                    break
            
            full_text = self.tokenizer.decode(generated)
            response = full_text[len(prompt):].strip()
        
        self.model.train()
        return response

# -----------------------------
# Kernel-Based Instruction Tuning
# -----------------------------
def instruction_tune(
    model,
    tokenizer,
    lm_studio_client: LMStudioClient,
    device,
    num_rounds: int = 100,
    examples_per_round: int = 5,
    training_batch_size: int = 4,
    learning_rate: float = 1e-5,
    save_path: str = 'instruction_tuned_model.pt'
):
    """
    Instruction tune by conversing with LM Studio (kernel-based).
    """
    global checkpoint_path_global
    checkpoint_path_global = save_path
    
    print("\n" + "="*60)
    print("KERNEL-BASED INSTRUCTION TUNING")
    print("="*60 + "\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    data_generator = ConversationalDataGenerator(model, tokenizer, lm_studio_client, device)
    
    all_examples = []
    best_avg_rating = 0.0
    
    try:
        for round_num in range(1, num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"ROUND {round_num}/{num_rounds}")
            print(f"{'='*60}\n")
            
            # Generate new training examples
            round_examples = []
            kernels = random.sample(DISCUSSION_KERNELS, min(examples_per_round, len(DISCUSSION_KERNELS)))
            
            for kernel in kernels:
                example = data_generator.generate_training_example(kernel)
                if example:
                    round_examples.append(example)
                    all_examples.append(example)
                time.sleep(1)
            
            if not round_examples:
                print("No examples generated this round, skipping...")
                continue
            
            avg_rating = sum(ex['rating'] for ex in round_examples) / len(round_examples)
            print(f"\nRound {round_num} Average Rating: {avg_rating:.3f}")
            
            # Train on recent examples
            training_examples = all_examples[-20:] if len(all_examples) > 20 else all_examples
            
            print(f"\nTraining on {len(training_examples)} examples...")
            
            num_epochs = 3
            for epoch in range(num_epochs):
                random.shuffle(training_examples)
                epoch_loss = 0.0
                num_batches = 0
                
                for i in range(0, len(training_examples), training_batch_size):
                    batch = training_examples[i:i+training_batch_size]
                    
                    batch_loss = 0.0
                    batch_weight = 0.0
                    
                    for example in batch:
                        full_text = example['query'] + " " + example['expert_response']
                        tokens = tokenizer.encode(full_text)
                        
                        if len(tokens) < 10:
                            continue
                        
                        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
                        target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
                        
                        logits = model(input_ids)
                        
                        query_length = len(tokenizer.encode(example['query']))
                        response_start = max(0, query_length - 1)
                        
                        if response_start < len(tokens) - 1:
                            response_logits = logits[:, response_start:, :]
                            response_targets = target_ids[:, response_start:]
                            
                            loss = F.cross_entropy(
                                response_logits.reshape(-1, model.vocab_size),
                                response_targets.reshape(-1),
                                reduction='mean'
                            )
                            
                            weighted_loss = loss * example['loss_weight']
                            batch_loss += weighted_loss
                            batch_weight += example['loss_weight']
                    
                    if batch_weight > 0:
                        batch_loss = batch_loss / batch_weight
                        
                        optimizer.zero_grad()
                        batch_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        epoch_loss += batch_loss.item()
                        num_batches += 1
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
            
            if avg_rating > best_avg_rating:
                best_avg_rating = avg_rating
                torch.save(model.state_dict(), save_path)
                print(f"\n✨ New best average rating: {best_avg_rating:.3f} - Model saved!")
            
            print("\n" + "-"*60)
            print("TESTING MODEL PROGRESS:")
            print("-"*60)
            
            test_kernel = random.choice(DISCUSSION_KERNELS)
            test_response = data_generator._generate_from_your_model(test_kernel)
            print(f"Query: {test_kernel}")
            print(f"Response: {test_response}\n")
    
    except KeyboardInterrupt:
        print("\n\nInstruction tuning interrupted by user.")
        print("Saving final model...")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    print("\n" + "="*60)
    print("INSTRUCTION TUNING COMPLETE!")
    print(f"Best Average Rating: {best_avg_rating:.3f}")
    print(f"Total Examples Generated: {len(all_examples)}")
    print(f"Model saved to: {save_path}")
    print("="*60 + "\n")

def conversational_instruction_tune(
    model,
    tokenizer,
    lm_studio_client: LMStudioClient,
    device,
    num_rounds: int = None,
    conversations_per_round: int = 3,
    turns_per_conversation: int = 5,
    training_batch_size: int = 4,
    learning_rate: float = 1e-5,
    save_path: str = 'conversational_tuned_model.pt',
    continuous: bool = False,
    max_seq_len: int = 512,
    stride: int = 256  # Overlap between chunks for context
):
    """
    Instruction tune through multi-turn conversations with LM Studio.
    
    Args:
        continuous: If True, run indefinitely until interrupted
        max_seq_len: Maximum sequence length for the model
        stride: Number of tokens to shift when creating overlapping windows
    """
    global checkpoint_path_global
    checkpoint_path_global = save_path
    
    print("\n" + "="*80)
    print("CONVERSATIONAL INSTRUCTION TUNING")
    if continuous:
        print("MODE: CONTINUOUS (will run until interrupted)")
    else:
        print(f"MODE: FIXED ({num_rounds} rounds)")
    print("="*80)
    print(f"Format: Multi-turn conversations between your AI and LM Studio")
    print(f"Conversations per round: {conversations_per_round}")
    print(f"Turns per conversation: {turns_per_conversation}")
    print(f"Max sequence length: {max_seq_len} tokens")
    print(f"Window stride: {stride} tokens (for overlapping chunks)")
    if not continuous:
        print(f"Total conversations planned: {num_rounds * conversations_per_round}")
    print("="*80 + "\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    conversation_manager = ConversationManager(model, tokenizer, lm_studio_client, device)
    
    all_conversations = []
    all_training_examples = []
    best_avg_rating = 0.0
    
    round_num = 0
    
    try:
        while True:
            round_num += 1
            
            # Check if we should stop (if not continuous mode)
            if not continuous and num_rounds is not None and round_num > num_rounds:
                break
            
            print(f"\n{'='*80}")
            if continuous:
                print(f"ROUND {round_num} (Continuous Mode)")
            else:
                print(f"ROUND {round_num}/{num_rounds}")
            print(f"{'='*80}\n")
            
            round_conversations = []
            round_examples = []
            
            # Generate topics - mix of preset and AI-generated
            topics = []
            for i in range(conversations_per_round):
                if random.random() < 0.3:  # 30% chance to use AI-generated topic
                    print(f"🎲 Generating random topic from LM Studio...")
                    topic = lm_studio_client.generate_conversation_topic()
                    print(f"   Generated: {topic}\n")
                    topics.append(topic)
                else:
                    topics.append(random.choice(CONVERSATION_STARTERS))
            
            for conv_num, topic in enumerate(topics, 1):
                print(f"\n🗣️  CONVERSATION {conv_num}/{conversations_per_round}")
                
                your_model_starts = (conv_num % 2 == 0)
                
                conversation_result = conversation_manager.have_conversation(
                    starter_prompt=topic,
                    num_turns=turns_per_conversation,
                    your_model_starts=your_model_starts
                )
                
                round_conversations.append(conversation_result)
                round_examples.extend(conversation_result['training_examples'])
                all_training_examples.extend(conversation_result['training_examples'])
                
                time.sleep(2)
            
            all_conversations.extend(round_conversations)
            
            if round_examples:
                avg_rating = sum(ex['rating'] for ex in round_examples) / len(round_examples)
                print(f"\n{'='*80}")
                print(f"ROUND {round_num} SUMMARY")
                print(f"Conversations: {len(round_conversations)}")
                print(f"Training examples: {len(round_examples)}")
                print(f"Average rating: {avg_rating:.3f}/1.0")
                print(f"{'='*80}\n")
            else:
                print("No examples generated this round.")
                continue
            
            print("🎓 Training on conversation examples...\n")
            
            training_pool = all_training_examples[-50:] if len(all_training_examples) > 50 else all_training_examples
            
            num_epochs = 3
            for epoch in range(num_epochs):
                random.shuffle(training_pool)
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_start in range(0, len(training_pool), training_batch_size):
                    batch = training_pool[batch_start:batch_start+training_batch_size]
                    
                    batch_loss = 0.0
                    batch_weight = 0.0
                    
                    for example in batch:
                        full_text = example['context'] + example['expert_response']
                        tokens = tokenizer.encode(full_text)
                        
                        if len(tokens) < 10:
                            continue
                        
                        context_tokens = tokenizer.encode(example['context'])
                        context_len = len(context_tokens)
                        
                        # Split into overlapping windows
                        num_windows = 0
                        window_loss = 0.0
                        
                        for start_idx in range(0, len(tokens) - 1, stride):
                            end_idx = min(start_idx + max_seq_len, len(tokens))
                            window_tokens = tokens[start_idx:end_idx]
                            
                            if len(window_tokens) < 2:
                                continue
                            
                            input_ids = torch.tensor([window_tokens[:-1]], dtype=torch.long, device=device)
                            target_ids = torch.tensor([window_tokens[1:]], dtype=torch.long, device=device)
                            
                            logits = model(input_ids)
                            
                            # Calculate which part of this window is the response
                            # (only train on response tokens, not context)
                            window_context_end = max(0, context_len - start_idx)
                            
                            if window_context_end < len(window_tokens) - 1:
                                # This window contains response tokens
                                response_start = window_context_end
                                response_logits = logits[:, response_start:, :]
                                response_targets = target_ids[:, response_start:]
                                
                                if response_targets.numel() == 0:
                                    continue
                                
                                loss = F.cross_entropy(
                                    response_logits.reshape(-1, model.vocab_size),
                                    response_targets.reshape(-1),
                                    reduction='mean'
                                )
                                
                                window_loss += loss
                                num_windows += 1
                            
                            # Stop if we've covered the whole sequence
                            if end_idx >= len(tokens):
                                break
                        
                        if num_windows > 0:
                            # Average loss across all windows for this example
                            avg_window_loss = window_loss / num_windows
                            weighted_loss = avg_window_loss * example['loss_weight']
                            batch_loss += weighted_loss
                            batch_weight += example['loss_weight']
                    
                    if batch_weight > 0:
                        batch_loss = batch_loss / batch_weight
                        
                        optimizer.zero_grad()
                        batch_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        epoch_loss += batch_loss.item()
                        num_batches += 1
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
            
            print(f"\n✅ Training complete for round {round_num}")
            
            # Save checkpoint after every round
            torch.save(model.state_dict(), save_path)
            print(f"💾 Checkpoint saved to {save_path}")
            
            if avg_rating > best_avg_rating:
                best_avg_rating = avg_rating
                print(f"✨ New best rating: {best_avg_rating:.3f}\n")
            
            print("-"*80)
            print("🧪 Quick Test:")
            print("-"*80)
            test_prompt = random.choice(["Hello! How are you?", "What's your favorite topic?", "Tell me something interesting."])
            test_history = [{'role': 'user', 'content': test_prompt}]
            test_response = conversation_manager.generate_from_your_model(test_history)
            print(f"Q: {test_prompt}")
            print(f"A: {test_response[:300]}...\n")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving model...")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    print("\n" + "="*80)
    print("CONVERSATIONAL INSTRUCTION TUNING COMPLETE!")
    print(f"Total rounds: {round_num}")
    print(f"Total conversations: {len(all_conversations)}")
    print(f"Total training examples: {len(all_training_examples)}")
    print(f"Best average rating: {best_avg_rating:.3f}")
    print(f"Model saved to: {save_path}")
    print("="*80 + "\n")
    
    return all_conversations

# -----------------------------
# Training utilities
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

# -----------------------------
# Chat mode function
# -----------------------------
def chat_mode(model, device, tokenizer):
    """Interactive chat mode with the model"""
    model.eval()
    print("\n" + "="*60)
    print("CHAT MODE - Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    try:
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Exiting chat mode...")
                    break
                
                if not user_input:
                    continue
                
                input_ids = tokenizer.encode(user_input)
                generated = input_ids.copy()
                
                with torch.no_grad():
                    for i in range(500):
                        context = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
                        logits = model(context)
                        logits = logits[0, -1] / 1.0
                        
                        if torch.isnan(logits).any():
                            break
                        
                        probs = F.softmax(logits, dim=0)
                        next_token = torch.multinomial(probs, 1).item()
                        generated.append(next_token)
                        
                        if next_token == tokenizer.eos_token_id:
                            break
                        
                        if i > 50 and len(generated) >= 2:
                            decoded_so_far = tokenizer.decode(generated[-10:])
                            if any(end in decoded_so_far for end in ['. ', '.\n', '! ', '!\n', '? ', '?\n']):
                                recent_text = tokenizer.decode(generated[-3:]).strip()
                                if recent_text and recent_text[-1] in '.!?':
                                    break
                
                full_text = tokenizer.decode(generated)
                response = full_text[len(user_input):].strip()
                
                print(f"Model: {response}\n")
                
            except KeyboardInterrupt:
                print("\nExiting chat mode...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    finally:
        model.train()

# -----------------------------
# Training on parquet files
# -----------------------------
def train_on_parquet_files(model, optimizer, scheduler, data_loader, device, vocab_size, 
                           batch_size, seq_len, eval_interval, best_loss, tokenizer):
    """
    Train on parquet files where each file is treated as an epoch.
    """
    total_files = data_loader.get_file_count()
    
    try:
        for epoch, (file_path, chunk_generator) in enumerate(data_loader.iterate_files(), 1):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/{total_files}: {Path(file_path).name}")
            print(f"{'='*60}")
            
            epoch_start_time = time.time()
            epoch_tokens = 0
            iteration = 0
            
            for chunk_data in chunk_generator:
                chunk_start_time = time.time()
                chunk_tokens = len(chunk_data)
                
                if chunk_tokens < seq_len + 1:
                    print(f"Chunk too small ({chunk_tokens} tokens), skipping...")
                    continue
                
                max_batches = max(1, (chunk_tokens - seq_len - 1) // (batch_size * seq_len))
                
                chunk_loss_sum = 0.0
                chunk_iterations = 0
                
                for _ in range(max_batches):
                    iteration += 1
                    
                    x_batch, y_batch = get_batch(chunk_data, batch_size, seq_len, device)
                    epoch_tokens += batch_size * seq_len
                    
                    logits = model(x_batch)
                    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
                    
                    if torch.isnan(loss) or loss.item() > 50:
                        print(f"Unstable loss at iteration {iteration}: {loss.item():.4f}")
                        print("Reinitializing model...")
                        model._init_weights()
                        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
                        continue
                    
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    
                    chunk_loss_sum += loss.item()
                    chunk_iterations += 1
                    
                    if iteration % eval_interval == 0:
                        elapsed = time.time() - epoch_start_time
                        tokens_per_sec = epoch_tokens / elapsed if elapsed > 0 else 0
                        
                        val_loss = estimate_loss(model, chunk_data, eval_iters=20, 
                                               batch_size=batch_size, seq_len=seq_len, 
                                               device=device, vocab_size=vocab_size)
                        
                        print(f"\n{'='*60}")
                        print(f"Epoch {epoch}/{total_files} | Iter {iteration}")
                        print(f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
                        print(f"Speed: {tokens_per_sec:.0f} tokens/sec | Time: {elapsed:.1f}s")
                        print(f"{'='*60}")
                        
                        if val_loss < best_loss:
                            best_loss = val_loss
                            print(f"✨ New best loss! {best_loss:.4f}")
                            torch.save(model.state_dict(), checkpoint_path_global)
                        
                        start_idx = chunk_data[0].item()
                        sample_text = sample(model, start_idx, max_len=300, temperature=0.8, device=device)
                        print(sample_text[:500])
                        print()
                
                chunk_time = time.time() - chunk_start_time
                avg_loss = chunk_loss_sum / chunk_iterations if chunk_iterations > 0 else 0
                
                print(f"Chunk processed: {chunk_tokens:,} tokens in {chunk_time:.1f}s | Avg Loss: {avg_loss:.4f}")
            
            epoch_time = time.time() - epoch_start_time
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} complete!")
            print(f"Tokens processed: {epoch_tokens:,}")
            print(f"Time: {epoch_time:.1f}s ({epoch_tokens/epoch_time:.0f} tokens/sec)")
            print(f"{'='*60}\n")
            
            torch.save(model.state_dict(), checkpoint_path_global)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving final model...")
        torch.save(model.state_dict(), checkpoint_path_global)
        print(f"Model saved to {checkpoint_path_global}")
    
    return best_loss

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Hyperparameters
    d_model = 384
    n_frequencies = 64
    n_layers = 4
    n_heads = 4
    batch_size = 32
    seq_len = 128
    learning_rate = 1e-6
    eval_interval = 1000
    
    # Initialize model
    model = HybridSpectralLM(vocab_size, d_model, n_frequencies, n_layers, n_heads).to(device)
    model_global = model
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: {n_layers} spectral layers + cross-attention")
    print(f"Model size: d_model={d_model}, n_heads={n_heads}\n")
    
    # Check for checkpoint
    checkpoint_path = 'best_model.pt'
    checkpoint_path_global = checkpoint_path
    best_loss = float('inf')
    
    if os.path.exists(checkpoint_path):
        print("="*60)
        print(f"Found checkpoint: {checkpoint_path}")
        print("="*60)
        print("\nOptions:")
        print("  1. Continue training from checkpoint")
        print("  2. Enter chat mode")
        print("  3. Start fresh training (ignore checkpoint)")
        print("  4. Instruction tune with LM Studio (kernel-based)")
        print("  5. 🗣️  Conversational learning with LM Studio (multi-turn)")
        
        while True:
            choice = input("\nEnter your choice (1/2/3/4/5): ").strip()
            
            if choice == '1':
                print(f"\nLoading checkpoint from {checkpoint_path}...")
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                print("Checkpoint loaded successfully! Continuing training...\n")
                break
            elif choice == '2':
                print(f"\nLoading checkpoint from {checkpoint_path}...")
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                print("Checkpoint loaded successfully!\n")
                chat_mode(model, device, tokenizer)
                sys.exit(0)
            elif choice == '3':
                print("\nStarting fresh training (checkpoint ignored)...\n")
                break
            elif choice == '4':
                print(f"\nLoading checkpoint from {checkpoint_path}...")
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                print("Checkpoint loaded successfully!\n")
                
                lm_studio_url = input("Enter LM Studio URL (default: http://192.168.1.91:1234): ").strip()
                if not lm_studio_url:
                    lm_studio_url = "http://192.168.1.91:1234"
                
                print(f"\nTesting connection to {lm_studio_url}...")
                lm_client = LMStudioClient(lm_studio_url)
                test_response = lm_client.generate_response("Hello", max_tokens=10)
                
                if test_response:
                    print(f"✓ Connection successful! Response: {test_response}\n")
                    
                    instruction_tune(
                        model=model,
                        tokenizer=tokenizer,
                        lm_studio_client=lm_client,
                        device=device,
                        num_rounds=50,
                        examples_per_round=5,
                        training_batch_size=4,
                        learning_rate=1e-5,
                        save_path='instruction_tuned_model.pt'
                    )
                    
                    print("\nEntering chat mode with instruction-tuned model...")
                    chat_mode(model, device, tokenizer)
                else:
                    print("✗ Connection failed! Please check LM Studio is running.")
                
                sys.exit(0)
            elif choice == '5':
                print(f"\nLoading checkpoint from {checkpoint_path}...")
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                print("Checkpoint loaded successfully!\n")
                
                lm_studio_url = input("Enter LM Studio URL (default: http://192.168.1.91:1234): ").strip()
                if not lm_studio_url:
                    lm_studio_url = "http://192.168.1.91:1234"
                
                print(f"\nTesting connection to {lm_studio_url}...")
                lm_client = LMStudioClient(lm_studio_url)
                test_response = lm_client.generate_response("Hello", max_tokens=10)
                
                if test_response:
                    print(f"✓ Connection successful! Response: {test_response}\n")
                    
                    # Run in continuous mode
                    conversations = conversational_instruction_tune(
                        model=model,
                        tokenizer=tokenizer,
                        lm_studio_client=lm_client,
                        device=device,
                        num_rounds=None,  # Not used in continuous mode
                        conversations_per_round=3,
                        turns_per_conversation=5,
                        training_batch_size=4,
                        learning_rate=1e-5,
                        save_path='conversational_tuned_model.pt',
                        continuous=True  # Run continuously
                    )
                    
                    print("\nEntering chat mode with conversationally-tuned model...")
                    chat_mode(model, device, tokenizer)
                else:
                    print("✗ Connection failed! Please check LM Studio is running.")
                
                sys.exit(0)
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
    
    # Ask user for training mode
    print("="*60)
    print("TRAINING MODE SELECTION")
    print("="*60)
    print("\nOptions:")
    print("  1. Train on text file (input.txt)")
    print("  2. Train on parquet files (folder with parquet files)")
    
    while True:
        mode = input("\nEnter your choice (1/2): ").strip()
        
        if mode == '1':
            print("\nLoading input.txt...")
            if not os.path.exists("input.txt"):
                print("Error: input.txt not found!")
                sys.exit(1)
            
            with open("input.txt", "r", encoding="utf-8") as f:
                text = f.read()
            
            data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
            print(f"Dataset size: {len(data):,} tokens\n")
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            num_iters = 100000
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters)
            
            if device == 'cuda':
                torch.backends.cudnn.benchmark = True
            
            start_time = time.time()
            tokens_processed = 0
            
            try:
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
                        
                        val_loss = estimate_loss(model, data, eval_iters=50, batch_size=batch_size, 
                                               seq_len=seq_len, device=device, vocab_size=vocab_size)
                        
                        print(f"\n{'='*60}")
                        print(f"Iter {iteration}/{num_iters}")
                        print(f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
                        print(f"Speed: {tokens_per_sec:.0f} tokens/sec | Time: {elapsed:.1f}s")
                        print(f"{'='*60}")
                        
                        if val_loss < best_loss:
                            best_loss = val_loss
                            print(f"✨ New best loss! {best_loss:.4f}")
                            torch.save(model.state_dict(), checkpoint_path)
                        
                        start_idx = data[0].item()
                        sample_text = sample(model, start_idx, max_len=300, temperature=0.8, device=device)
                        print(sample_text[:500])
                        print()
            
            except KeyboardInterrupt:
                print("\n\nTraining interrupted by user.")
                print("Saving final model...")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved to {checkpoint_path}")
            
            break
            
        elif mode == '2':
            folder_path = input("\nEnter path to parquet folder: ").strip()
            
            if not os.path.exists(folder_path):
                print(f"Error: Folder '{folder_path}' not found!")
                continue
            
            text_column = input("Enter the name of the text column: ").strip()
            
            try:
                data_loader = ParquetDataLoader(folder_path, text_column, tokenizer, chunk_size=10000)
            except Exception as e:
                print(f"Error loading parquet files: {e}")
                continue
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = None
            
            if device == 'cuda':
                torch.backends.cudnn.benchmark = True
            
            best_loss = train_on_parquet_files(
                model, optimizer, scheduler, data_loader, device, vocab_size,
                batch_size, seq_len, eval_interval, best_loss, tokenizer
            )
            
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Final model saved to: {checkpoint_path}")
    print("="*60)