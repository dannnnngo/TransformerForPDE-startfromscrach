import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# From Attention is All You Need paper
# First do the input embeddings.
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, input_size: int):
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.embedding = nn.Embedding(input_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # in the paper it is mentioned that the input embeddings are scaled by sqrt(d_model)

# seq_len is the length of the input sequence, dropout is to make the model less overfit.
# Then get the positional encoding to give other informations.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model) to store the positional encodings
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1))
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Calculate in the logspace 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # apply the sin in even position; cos in odd position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape to (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        # register the positional encoding as a buffer, so it is not a parameter of the model 
        # and will not be updated during training
        self.register_buffer('pe', pe) 

    def forward(self, x):
        # make the pe not trainable
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) 
        return self.dropout(x)

# Use the LayerNormalization to normalize the input embeddings and the output of the attention layer.
# General equation is like x_hat = (x_j - mean_j)/sqrt(epsilon + var_j^2) 
# -> epsilon to introduce some fluctuations ; -> if var_j = 0, then x_hat is undesirable.
class LayerNormalization(nn.Module):
    def __init__(self, epsilon:float = 1e-6, ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / torch.sqrt(var + self.epsilon) + self.bias
    
# Position-wise FeedForward Network
# In the paper they use: FFN(x) = max(0, x*W_1 + b_1)W_2 + b_2
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W_1 and b_1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W_2 and b_2
    
    def forward(self, x):
        # [Batch, seq_len, d_model] -> [Batch, seq_len, d_ff]
        return self.linear2(self.dropout(F.relu(self.linear1(x)))) # [Batch, seq_len, d_model]
    
# Multi-head attention 
# Input[seq,d_model] -> Q[seq,d_model] x WQ[d_model,d_model] = [seq,d_model], K x Wk, V x Wv
# Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))*V
# head_i = Attention(QW_iQ, KW_iK, VW_iV); MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) x Wo
# Wo is the final linear layer to project the concatenated heads back to d_model
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 # d_model is divisible by h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -2)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9) # mask the padding tokens
        attention_scores = attention_scores.softmax(dim=-1) # [Batch, h, seq_len, seq_len]

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores # [Batch, h, seq_len, d_k], [Batch, h, seq_len, seq_len]

    # sometimes we need to mask the input to the attention layer, for example in the decoder
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # [Batch, seq_len, d_model] -> [Batch, seq_len, d_model]
        key = self.w_k(k) # [Batch, seq_len, d_model] -> [Batch, seq_len, d_model]
        value = self.w_v(v) # [Batch, seq_len, d_model] -> [Batch, seq_len, d_model]

        # we do not want to split the sentence but split on the embeddings to the head part. 
        # [Batch, seq_len, d_model] -> [Batch, seq_len, h, d_k] -> [Batch, h, seq_len, d_k]
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # [Batch, h, seq_len, d_k] -> [Batch, seq_len, h, d_k] -> [Batch, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # [Batch, seq_len, d_model] -> [Batch, seq_len, d_model]
        return self.w_o(x)

# Residual Connection
class ResidualConnection(nn.Module):
    def __init__(self, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # [Batch, seq_len, d_model] + [Batch, seq_len, d_model]

# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # use module list 
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    # No padding words interact with other word
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# Nx layer number of Encoder
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
# Ny layer number of Decoder
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
# Linear Layer
# Map the d_model to the target size (projecting the embeding to the target size)
class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # [Batch, seq_len, d_model] -> [Batch, seq_len, target_size]
        return torch.log_softmax(self.proj(x), dim=-1)

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, 
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encoder(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    def decoder(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_size:int, tgt_size:int, 
                      src_seq_len:int, tgt_seq_len:int, 
                      d_model:int=512, N:int=6, dropout:float=0.1, h: int = 8, d_ff:int=2048) -> Transformer:

    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_size)
    tgt_embed = InputEmbeddings(d_model, tgt_size)
    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    # Create the projection layer

    # For Encoder Blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # For Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_size)

    # Create the transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the Parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) # Xavier initialization
        else:
            nn.init.constant_(p, 0) # Constant initialization

    return transformer