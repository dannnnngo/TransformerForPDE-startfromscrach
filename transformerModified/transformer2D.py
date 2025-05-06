import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding2D(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim):
        super().__init__()
        self.patch_size = patch_size  # (h, w)
        self.proj = nn.Conv2d(
            in_channels, emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, emb_dim, H', W']
        B, emb_dim, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, emb_dim]
        return x, (H, W)

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h, max_w):
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        # Pre-compute position encodings
        self.register_buffer("pos_embed", self._get_positional_encoding(max_h, max_w, d_model))
        
    def _get_positional_encoding(self, max_h, max_w, d_model):
        pe = torch.zeros(max_h, max_w, d_model)
        # The dimension for each axis (x and y) is half of d_model
        axis_dim = d_model // 2
        # Create position meshgrid
        y_pos = torch.arange(max_h).unsqueeze(1).float()  # [max_h, 1]
        x_pos = torch.arange(max_w).unsqueeze(0).float()  # [1, max_w]
        # For each dimension i
        for i in range(axis_dim // 2):
            div_term_x = torch.pow(torch.tensor(10000.0), torch.tensor(4.0 * i / axis_dim))
            div_term_y = torch.pow(torch.tensor(10000.0), torch.tensor(4.0 * i / axis_dim))

            # PE(x,y,2i) = sin(x/10000^(4i/D))
            pe[:, :, 2*i] = torch.sin(x_pos / div_term_x)
            # PE(x,y,2i+1) = cos(x/10000^(4i/D))
            pe[:, :, 2*i + 1] = torch.cos(x_pos / div_term_x)
            # PE(x,y,2j+D/2) = sin(y/10000^(4j/D))
            pe[:, :, 2*i + axis_dim] = torch.sin(y_pos / div_term_y)
            # PE(x,y,2j+1+D/2) = cos(y/10000^(4j/D))
            pe[:, :, 2*i + 1 + axis_dim] = torch.cos(y_pos / div_term_y)
        
        # Reshape to [max_h*max_w, d_model]
        pe = pe.view(-1, d_model)
        return pe
        
    def forward(self, x, H, W):
        # Get the appropriate slice of positional encodings
        # First, create a mask for the positions we need
        h_indices = torch.arange(H, device=x.device).view(-1, 1).repeat(1, W).view(-1)
        w_indices = torch.arange(W, device=x.device).repeat(H)
        position_indices = h_indices * self.max_w + w_indices
        
        position_indices = torch.tensor(position_indices, device=x.device)
        pe = self.pos_embed[position_indices]  # Shape: [H*W, d_model]

        # Add positional encoding to input
        # pe has shape [H*W, d_model], x has shape [batch_size, H*W, d_model]
        return x + pe.unsqueeze(0)  # Broadcasting handles batch dimension


# Use PyTorch's LayerNorm
class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, dropout, d_ff):
        super().__init__()
        # Use PyTorch's LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multi-head attention
        self.self_attention = nn.MultiheadAttention(
            d_model, h, dropout=dropout, batch_first=True
        )
        
        # Feed forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, src_mask=None):
        # Pre-norm architecture (more stable training)
        normed_x = self.norm1(x)
        attn_output, _ = self.self_attention(
            normed_x, normed_x, normed_x, 
            key_padding_mask=src_mask
        )
        x = x + attn_output
        
        x = x + self.ff(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, h, dropout, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.self_attention = nn.MultiheadAttention(
            d_model, h, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, h, dropout=dropout, batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self attention
        normed_x = self.norm1(x)
        attn_output, _ = self.self_attention(
            normed_x, normed_x, normed_x,
            key_padding_mask=tgt_mask
        )
        x = x + attn_output
        
        # Cross attention
        normed_x = self.norm2(x)
        attn_output, _ = self.cross_attention(
            normed_x, encoder_output, encoder_output,
            key_padding_mask=src_mask
        )
        x = x + attn_output
        
        # Feed forward
        x = x + self.ff(self.norm3(x))
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, h, dropout, d_ff, N):
        super().__init__()
        # Create all encoder layers at once
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, h, dropout, d_ff) for _ in range(N)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, d_model, h, dropout, d_ff, N):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, h, dropout, d_ff) for _ in range(N)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer2D(nn.Module):
    def __init__(self, d_model, out_channels, out_shape):
        super().__init__()
        self.out_channels = out_channels
        self.out_shape = out_shape  # (H, W)
        self.proj = nn.Linear(d_model, out_channels)
    
    def forward(self, x):
        # x: [B, num_patches, d_model]
        B = x.size(0)
        out = self.proj(x)  # [B, num_patches, out_channels]
        H, W = self.out_shape
        out = out.transpose(1, 2).reshape(B, self.out_channels, H, W)
        return out

class OptimizedTransformer2D(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        input_shape,  # (H, W)
        patch_size=(1, 1),
        d_model=128, 
        N=2, 
        dropout=0.1, 
        h=4, 
        d_ff=256
    ):
        super().__init__()
        # Patch embedding
        self.patch_embed = PatchEmbedding2D(in_channels, patch_size, d_model)
        
        # Calculate output shapes after patch embedding
        H, W = (
            input_shape[0] // patch_size[0],
            input_shape[1] // patch_size[1]
        )
        
        # Positional encoding
        self.pos_enc = PositionalEncoding2D(d_model, H, W)
        
        # Encoder and decoder
        self.encoder = Encoder(d_model, h, dropout, d_ff, N)
        self.decoder = Decoder(d_model, h, dropout, d_ff, N)
        
        # Projection layer
        self.projection_layer = ProjectionLayer2D(d_model, out_channels, (H, W))
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        # src: [B, C, H, W]
        src_embed, (H, W) = self.patch_embed(src)
        src_embed = self.pos_enc(src_embed, H, W)
        
        # Encode
        memory = self.encoder(src_embed, src_mask)
        
        # Decode (if target provided)
        if tgt is not None:
            tgt_embed, _ = self.patch_embed(tgt)
            tgt_embed = self.pos_enc(tgt_embed, H, W)
            out = self.decoder(tgt_embed, memory, src_mask, tgt_mask)
        else:
            out = memory
        
        # Project to output shape
        return self.projection_layer(out)

def build_optimized_transformer2d(
    in_channels=1, 
    out_channels=1,
    input_shape=(64, 64),  # (H, W)
    patch_size=(1, 1),
    d_model=128, 
    N=2, 
    dropout=0.1, 
    h=4, 
    d_ff=256
):
    return OptimizedTransformer2D(
        in_channels=in_channels,
        out_channels=out_channels,
        input_shape=input_shape,
        patch_size=patch_size,
        d_model=d_model,
        N=N,
        dropout=dropout,
        h=h,
        d_ff=d_ff
    )