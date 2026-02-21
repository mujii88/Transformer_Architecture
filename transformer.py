
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
  def __init__(self,d_model,vocab_size):
    super().__init__()
    self.vocab_size=vocab_size
    self.d_model=d_model
    self.embeddings=nn.Embedding(vocab_size,d_model)

  def forward(self,x):
    return self.embeddings(x)*torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

# sentence='the animal is too tired'
# vocab_size=len(sentence.split(' '))
# m=InputEmbeddings(10,vocab_size)
# tokens=sentence.split(' ')

# word_to_id = {word: i for i, word in enumerate(tokens)}
# input_indices = torch.tensor([word_to_id[w] for w in tokens], dtype=torch.long)
# embeddings = m(input_indices)
# len(embeddings)
# input_indices
# embeddings

import math
class Positional_Encoding(nn.Module):
  def __init__(self,d_model,max_len,dropout=0.1):
    super().__init__()
    self.d_model=d_model
    self.seq_len=max_len
    self.dropout=nn.Dropout(dropout)


    pos_matrix=torch.zeros(self.seq_len,self.d_model)

    position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)

    div_term=torch.exp(
        torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model)
    )

    pos_matrix[:,0::2]=torch.sin(position*div_term)
    pos_matrix[:,1::2]=torch.cos(position*div_term)

    pos_matrix=pos_matrix.unsqueeze(0)
    self.register_buffer('pos_matrix',pos_matrix)


  def forward(self,x):
    x=x+(self.pos_matrix[:,:x.shape[1],:]).requires_grad_(False)

    return self.dropout(x)

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads

        assert d_model % heads == 0

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = scores.softmax(dim=-1)

        if dropout is not None:
            weights = dropout(weights)

        return (weights @ value), weights

    def forward(self, q, k, v, mask=None):
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        query = query.view(query.shape[0], -1, self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).transpose(1, 2)

        x, self.attention_weights = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        return self.output(x)

# --- Test Example ---

# d_model = 512
# heads = 8
# seq_len = 10
# batch_size = 2

# mha = MultiHeadAttention(d_model, heads)
# x = torch.randn(batch_size, seq_len, d_model)
# mask = torch.ones(batch_size, 1, 1, seq_len)

# out = mha(x, x, x, mask)
# print(out.shape)

import torch
import torch.nn as nn

a=nn.Linear(3,2)

import torch
import torch.nn as nn


class PositionFeedForward(nn.Module):
  def __init__(self,d_model,d_diff,dropout=0.1):
    super().__init__()
    self.w1=nn.Linear(d_model,d_diff)


    self.w2=nn.Linear(d_diff,d_model)

    self.dropout=nn.Dropout(dropout)

    self.relu=nn.ReLU()


  def forward(self,x):

    return self.w2(self.dropout(self.relu(self.w1(x))))

import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    # 1. We add 'features' (d_model) so we can size our parameters correctly
    def __init__(self, features: int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps

        # 2. We make alpha and bias a vector of size 512!
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # Pro-tip: Standard LayerNorm doesn't use Bessel's correction (unbiased=False)
        std = x.std(dim=-1, unbiased=False, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias

import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()

        # 1. Self-Attention Sub-layer
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.norm1 = LayerNormalization(d_model) # Using PyTorch's built-in LayerNorm for stability
        self.dropout1 = nn.Dropout(dropout)

        # 2. Feed-Forward Sub-layer
        self.feed_forward = PositionFeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNormalization(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x shape: (Batch, Seq_Len, d_model)
        # mask shape: (Batch, 1, 1, Seq_Len)

        # --- Sub-layer 1: Attention ---
        # 1. Calculate Attention
        # Note: In Encoder, Query=Key=Value=x
        attention_out = self.self_attention(x, x, x, mask)

        # 2. Add & Norm
        # We add the ORIGINAL input 'x' to the attention output (Residual Connection)
        x = self.norm1(x + self.dropout1(attention_out))

        # --- Sub-layer 2: Feed Forward ---
        # 3. Calculate Feed Forward
        ff_out = self.feed_forward(x)

        # 4. Add & Norm
        # We add the input of this sub-layer (which is the output of norm1) to the FF output
        x = self.norm2(x + self.dropout2(ff_out))

        return x

import torch
import torch.nn as nn
import copy

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.N = N

        # 1. Embeddings & Positional Encoding
        self.embed = InputEmbeddings(d_model, vocab_size)
        self.pe = Positional_Encoding(d_model, max_len, dropout)

        # 2. The Stack of N Encoder Layers
        # We use nn.ModuleList to store a list of layers
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, heads, d_ff, dropout)
            for _ in range(N)
        ])

        # 3. Final Normalization Layer
        self.norm = LayerNormalization(d_model)

    def forward(self, src, mask):
        # src shape: (Batch, Seq_Len)

        # 1. Embed and Add Position
        x = self.embed(src) # (Batch, Seq_Len) -> (Batch, Seq_Len, d_model)
        x = self.pe(x)

        # 2. Pass through each of the N layers
        for layer in self.layers:
            x = layer(x, mask)

        # 3. Final Norm
        return self.norm(x)

"""# **Decoder**"""

import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()

        # Instantiate the layers inside the block (The Modern Way!)
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, heads, dropout)
        self.feed_forward = PositionFeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. Masked Self-Attention
        _x = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(_x))

        # 2. Cross-Attention (Query from Decoder 'x', Key/Value from 'encoder_output')
        _x = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(_x))

        # 3. Feed Forward
        _x = self.feed_forward(x)
        x = self.norm3(x + self.dropout(_x))

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.N = N

        # 1. Embeddings & Positional Encoding for Target Language
        self.embed = InputEmbeddings(d_model, vocab_size)
        self.pe = Positional_Encoding(d_model, max_len, dropout)

        # 2. The Stack of N Decoder Layers
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, heads, d_ff, dropout)
            for _ in range(N)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, encoder_output, src_mask, tgt_mask):
        # 1. Embed and Add Position
        x = self.embed(tgt)
        x = self.pe(x)

        # 2. Pass through each of the N layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, N=6, heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        # Build the entire Encoder stack
        self.encoder = TransformerEncoder(src_vocab_size, d_model, N, heads, d_ff, dropout)

        # Build the entire Decoder stack
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, N, heads, d_ff, dropout)

        # Build the final word generator
        self.generator = Generator(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Pass input through Encoder
        encoder_output = self.encoder(src, src_mask)

        # Pass target and encoder output through Decoder
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        # Generate word probabilities
        return self.generator(decoder_output)