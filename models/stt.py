import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, tokens_per_frame, hidden_dim)
        batch_size, seq_len, tokens_per_frame, hidden_dim = x.shape
        
        x_flat = x.reshape(batch_size * seq_len, tokens_per_frame, hidden_dim)
        
        attn_output, _ = self.mha(x_flat, x_flat, x_flat)
        attn_output = self.ln(attn_output + x_flat)
        
        attn_output = attn_output.reshape(batch_size, seq_len, tokens_per_frame, hidden_dim)
        return attn_output

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.pe = PositionalEncoding(hidden_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, tokens_per_frame, hidden_dim)
        batch_size, seq_len, tokens_per_frame, hidden_dim = x.shape
        
        outputs = []
        for i in range(tokens_per_frame):
            token_seq = x[:, :, i, :]  # (batch, seq_len, hidden_dim)
            
            token_seq = self.pe(token_seq)
            
            attn_output, _ = self.mha(token_seq, token_seq, token_seq)
            attn_output = self.ln(attn_output + token_seq)
            
            outputs.append(attn_output.unsqueeze(2))
        
        return torch.cat(outputs, dim=2)

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim=None):
        super().__init__()
        if ff_dim is None:
            ff_dim = hidden_dim * 4
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x: (batch, seq_len, tokens_per_frame, hidden_dim)
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.ln(x + residual)
        return x

class SpatiotemporalTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.spatial_attn = SpatialAttention(hidden_dim, num_heads)
        self.temporal_attn = TemporalAttention(hidden_dim, num_heads)
        self.ff = FeedForward(hidden_dim)
        
    def forward(self, x):
        x = self.spatial_attn(x)
        x = self.temporal_attn(x)
        x = self.ff(x)
        return x

class SpatiotemporalTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            SpatiotemporalTransformerBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x