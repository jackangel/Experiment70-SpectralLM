# Hybrid Spectral-Attention Language Model

## Architecture Overview

This is a character-level language model that combines **spectral processing** with **cross-attention mechanisms** to generate text. The architecture introduces a novel approach by processing sequences through spectral transformations while maintaining attention-based connections between layers.

## Core Components

### 1. **Spectral Layer with Cross-Attention**

The fundamental building block that replaces traditional transformer layers. Each layer performs three main operations:

#### Spectral Processing
- Projects input embeddings into a **spectral space** with complex-valued representations (real and imaginary components)
- Applies **recurrent spectral dynamics** with learnable decay rates and rotation frequencies
- Each timestep applies:
  - **Decay**: Exponential damping controlled by learned parameters
  - **Rotation**: Complex rotation in the spectral domain using learned frequencies
  - **Accumulation**: Integration of new input with decayed state
- Projects spectral representations back to embedding space

The spectral processing acts as a recurrent mechanism that captures temporal dependencies through continuous-time dynamics rather than discrete attention weights.

#### Cross-Attention
- Connects each layer to the **previous layer's output** (skipped for the first layer)
- Uses standard multi-head attention where:
  - **Query** comes from the current layer
  - **Key and Value** come from the previous layer's output
  - Applies causal masking to prevent attending to future tokens
- Enables information flow across layers with explicit inter-layer dependencies

#### Feed-Forward Network
- Standard transformer FFN with GELU activation
- Expands to 4× model dimension and projects back
- Includes dropout for regularization

All three components use **residual connections** with small scaling factors (0.1) for training stability.

### 2. **Model Architecture**

```
Input Tokens
    ↓
Token Embedding + Positional Embedding
    ↓
┌─────────────────────────────────────┐
│ Layer 1: Spectral Processing        │
│          + FFN                       │
│          (no cross-attention)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Layer 2: Spectral Processing        │
│          + Cross-Attention (Layer 1) │
│          + FFN                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Layer 3: Spectral Processing        │
│          + Cross-Attention (Layer 2) │
│          + FFN                       │
└─────────────────────────────────────┘
    ↓
    ... (continues for n_layers)
    ↓
Layer Normalization
    ↓
Output Projection (tied with input embeddings)
    ↓
Logits over Vocabulary
```

### 3. **Key Design Choices**

#### Spectral State Evolution
The spectral state evolves according to:
- **State decay**: `state = state × σ(log_decay)`
- **Input integration**: `state = state + input × 0.1`
- **Complex rotation**: 
  - `new_real = state_real × cos(ω) - state_imag × sin(ω)`
  - `new_imag = state_real × sin(ω) + state_imag × cos(ω)`
- **Clamping**: States are clamped to [-10, 10] for numerical stability

#### Cross-Layer Information Flow
Unlike standard transformers where layers operate independently:
- Each layer receives the **full output** of the previous layer
- Cross-attention allows selective retrieval of relevant information from earlier processing stages
- Creates a **hierarchical representation** where deeper layers can query shallower ones

#### Weight Tying
The output projection matrix shares weights with the input embedding matrix, reducing parameters and enforcing consistency between input and output spaces.

## Model Dimensions

Default configuration:
- **d_model**: 384 (embedding dimension)
- **n_frequencies**: 64 (spectral state dimension)
- **n_layers**: 4 (number of spectral layers)
- **n_heads**: 4 (attention heads in cross-attention)
- **context_length**: 512 tokens (maximum sequence length)

## Training Features

### Stability Mechanisms
- **Gradient clipping** (max norm = 1.0)
- **Small residual scaling** (0.1×) for all skip connections
- **State clamping** in spectral processing
- **Xavier initialization** with reduced gains
- **Loss monitoring** with automatic reinitialization on NaN/explosion

### Optimization
- **AdamW optimizer** with weight decay (0.01)
- **Cosine annealing** learning rate schedule
- **Initial learning rate**: 3e-4
- **Training sequence length**: 128 tokens
- **Batch size**: 32

### Checkpointing
- Saves best model based on validation loss
- Supports resuming training from checkpoints
- Includes interactive chat mode for inference

## Novel Aspects

1. **Spectral Recurrence**: Replaces self-attention with continuous-time spectral dynamics
2. **Cross-Layer Attention**: Explicit attention between adjacent layers (not within layers)
3. **Hybrid Design**: Combines recurrent spectral processing with attention mechanisms
4. **Complex-Valued States**: Uses complex rotations in spectral domain for richer dynamics

This architecture aims to capture both local temporal dependencies (through spectral recurrence) and long-range dependencies (through cross-layer attention) while being more parameter-efficient than pure transformer models.
