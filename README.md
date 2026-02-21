# ⚙️ Transformer Architecture: Pure PyTorch Implementation

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-Architecture-blue)
![Status](https://img.shields.io/badge/Status-Complete-success)

A complete, from-scratch PyTorch implementation of the neural network architecture described in the seminal paper *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)*. 

This repository contains the pure structural blueprint of the Transformer model. It bypasses high-level wrapper libraries (like `nn.Transformer`) to implement the mathematical core natively, including custom Attention matrices, Positional Encodings, and Encoder/Decoder stacks. 



## 🛠️ Architectural Components

The codebase is modular, breaking down the complex Transformer into its fundamental PyTorch classes:

1. **`InputEmbeddings`**: Projects discrete integer tokens into a $d_{model}$-dimensional continuous vector space, scaled by $\sqrt{d_{model}}$.
2. **`Positional_Encoding`**: Injects non-learned sinusoidal frequencies (Sine and Cosine) into the embeddings to provide sequence order context without relying on recurrence.
3. **`MultiHeadAttention`**: Implements the core $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$ calculus, supporting parallel attention heads and custom masking.
4. **`PositionFeedForward`**: The fully connected, position-wise feed-forward network with a ReLU activation and dropout for regularization.
5. **`LayerNormalization`**: A custom implementation of LayerNorm to stabilize the hidden state dynamics across the deep residual connections.
6. **`EncoderBlock` & `DecoderBlock`**: The stacked sub-layers handling Self-Attention and Cross-Attention routing.
7. **`Generator`**: The final linear projection and `log_softmax` layer mapping the continuous decoder output back to discrete vocabulary probabilities.

## 🚀 Usage / Instantiation

Because this is a pure architectural implementation, you can instantiate the `Transformer` class and pass dummy tensors through it to verify the forward-pass math and tensor shapes.

```python
import torch
from model import Transformer

# 1. Define hyperparameters
src_vocab_size = 10000
tgt_vocab_size = 10000
d_model = 512
max_seq_len = 100
batch_size = 8

# 2. Initialize the architecture
model = Transformer(
    src_vocab_size=src_vocab_size, 
    tgt_vocab_size=tgt_vocab_size, 
    d_model=d_model, 
    N=6,          # Number of Encoder/Decoder layers
    heads=8,      # Number of attention heads
    d_ff=2048,    # Feed-forward hidden dimension
    dropout=0.1
)

# 3. Create dummy data and masks
src_dummy = torch.randint(0, src_vocab_size, (batch_size, max_seq_len))
tgt_dummy = torch.randint(0, tgt_vocab_size, (batch_size, max_seq_len))
src_mask = torch.ones(batch_size, 1, 1, max_seq_len)
tgt_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)

# 4. Execute the Forward Pass
output = model(src_dummy, tgt_dummy, src_mask, tgt_mask)

print(f"Output Shape: {output.shape}") 
# Expected: [Batch, Seq_Len, Tgt_Vocab_Size] -> [8, 100, 10000]