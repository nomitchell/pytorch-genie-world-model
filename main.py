import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import gym
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
class Config:
    # General
    seed = 42
    image_size = 64
    batch_size = 16
    gradient_accumulation_steps = 4
    num_workers = 4
    
    # Video Tokenizer
    vq_embedding_dim = 64 
    vq_num_embeddings = 512 
    vq_commitment_cost = 0.25
    
    # Spatiotemporal Transformer, Reduced from original paper
    hidden_dim = 256  # 512
    num_heads = 4  # 8
    num_layers = 4  # 12? 
    
    # Latent Action Model
    num_actions = 14 
    history_length = 16 
    
    # Dynamics Model
    dynamics_context_frames = 16 # TODO make naming more consiste
    
    # Training
    epochs = 30
    lr = 3e-4
    lr_min = 1e-5
    weight_decay = 1e-5
    warmup_epochs = 5

config = Config()

# Set random seeds for reproducibility
torch.manual_seed(config.seed)
np.random.seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)

# Video Tokenizer

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.resblock(x)
        shortcut = self.shortcut(x)
        return self.relu(residual + shortcut)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, latent_dim=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        
        # Downsampling layers - 64x64 -> 32x32 -> 16x16 -> 8x8
        self.down1 = nn.Sequential(
            Residual(hidden_dim, hidden_dim),
            DownSample(hidden_dim, hidden_dim * 2)
        )
        self.down2 = nn.Sequential(
            Residual(hidden_dim * 2, hidden_dim * 2),
            DownSample(hidden_dim * 2, hidden_dim * 4)
        )
        self.down3 = nn.Sequential(
            Residual(hidden_dim * 4, hidden_dim * 4),
            DownSample(hidden_dim * 4, hidden_dim * 4)
        )
        
        self.residual = Residual(hidden_dim * 4, hidden_dim * 4)
        self.conv_out = nn.Conv2d(hidden_dim * 4, latent_dim, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.residual(x)
        x = self.conv_out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dim=64, latent_dim=64):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_dim, hidden_dim * 4, 1)
        
        # Upsampling layers - 8x8 -> 16x16 -> 32x32 -> 64x64
        self.up1 = nn.Sequential(
            Residual(hidden_dim * 4, hidden_dim * 4),
            UpSample(hidden_dim * 4, hidden_dim * 2)
        )
        self.up2 = nn.Sequential(
            Residual(hidden_dim * 2, hidden_dim * 2),
            UpSample(hidden_dim * 2, hidden_dim)
        )
        self.up3 = nn.Sequential(
            Residual(hidden_dim, hidden_dim),
            UpSample(hidden_dim, hidden_dim)
        )
        
        self.residual = Residual(hidden_dim, hidden_dim)
        self.conv_out = nn.Conv2d(hidden_dim, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.residual(x)
        x = self.conv_out(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, x):
        # BCHW -> BHWC
        x = x.permute(0, 2, 3, 1).contiguous()
        input_shape = x.shape
        
        flat_input = x.view(-1, self.embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = x + (quantized - x).detach()
        
        # BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, loss, encoding_indices.view(x.shape[0], -1)

class VQVAEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(in_channels=3, hidden_dim=64, latent_dim=config.vq_embedding_dim)
        self.vq = VectorQuantizer(
            num_embeddings=config.vq_num_embeddings,
            embedding_dim=config.vq_embedding_dim,
            commitment_cost=config.vq_commitment_cost
        )
        self.decoder = Decoder(out_channels=3, hidden_dim=64, latent_dim=config.vq_embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, indices = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, indices

    def encode(self, x):
        z = self.encoder(x)
        quantized, _, indices = self.vq(z)
        
        # Ensure all indices are within bounds
        indices = torch.clamp(indices, 0, self.vq.num_embeddings - 1)
        return indices

    def decode(self, indices):
        batch_size = indices.shape[0]

        # Reshaping indices to match dimension expected
        indices = indices.view(batch_size, 8, 8)
        
        one_hot = torch.zeros(batch_size, self.vq.num_embeddings, 8, 8, device=indices.device)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)
        
        quantized = torch.matmul(one_hot.permute(0, 2, 3, 1), self.vq.embeddings.weight)
        quantized = quantized.permute(0, 3, 1, 2)
        
        x_recon = self.decoder(quantized)
        return x_recon

# Spatiotemporal Transformer

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

# Latent Action Model

class LatentActionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.history_length = config.history_length
        
        self.token_embedding = nn.Embedding(config.vq_num_embeddings, config.hidden_dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, config.hidden_dim))  # 8x8=64 tokens per frame
        
        self.encoder = SpatiotemporalTransformer(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers
        )
        
        self.to_latent_action = nn.Sequential(
            nn.Linear(config.hidden_dim * 64, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        self.decoder = SpatiotemporalTransformer(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers
        )
        
        self.to_logits = nn.Linear(config.hidden_dim, config.vq_num_embeddings)

    def forward(self, prev_frames_tokens, next_frame_tokens=None):
        batch_size = prev_frames_tokens.shape[0]
        history_length = prev_frames_tokens.shape[1]
        num_tokens = prev_frames_tokens.shape[2]
        
        # Ensure all token indices are within bounds for embedding layer
        prev_frames_tokens = torch.clamp(prev_frames_tokens, 0, self.token_embedding.num_embeddings - 1)
        if next_frame_tokens is not None:
            next_frame_tokens = torch.clamp(next_frame_tokens, 0, self.token_embedding.num_embeddings - 1)
        
        prev_embeds = []
        for i in range(history_length):
            embeds = self.token_embedding(prev_frames_tokens[:, i])  # (batch, num_tokens, hidden_dim)
            embeds = embeds + self.pos_embedding
            prev_embeds.append(embeds)
        
        # Stack all previous frames as a sequence
        transformer_input = torch.stack(prev_embeds, dim=1)  # (batch, history_length, num_tokens, hidden_dim)
        
        # If we have next frame tokens (for training), include them in the encoding
        if next_frame_tokens is not None:
            next_embeds = self.token_embedding(next_frame_tokens)  # (batch, num_tokens, hidden_dim)
            next_embeds = next_embeds + self.pos_embedding
            
            transformer_input = torch.cat([
                transformer_input, 
                next_embeds.unsqueeze(1)
            ], dim=1)  # (batch, history_length+1, num_tokens, hidden_dim)
        
        encoded = self.encoder(transformer_input)  # (batch, history_length(+1), num_tokens, hidden_dim)
        
        latent_actions = []
        for i in range(history_length):
            frame_encoding = encoded[:, i].reshape(batch_size, -1)  # (batch, num_tokens * hidden_dim)
            latent_action = self.to_latent_action(frame_encoding)  # (batch, hidden_dim)
            latent_actions.append(latent_action)
        
        latent_actions = torch.stack(latent_actions, dim=1)  # (batch, history_length, hidden_dim)
        
        if next_frame_tokens is not None:
            decoder_input = []
            for i in range(history_length):
                frame_embed = prev_embeds[i]
                action_embed = latent_actions[:, i].unsqueeze(1).expand(-1, num_tokens, -1)
                combined = frame_embed + action_embed
                decoder_input.append(combined)
            
            decoder_input = torch.stack(decoder_input, dim=1)  # (batch, history_length, num_tokens, hidden_dim)
            
            decoded = self.decoder(decoder_input)  # (batch, history_length, num_tokens, hidden_dim)
            
            next_frame_features = decoded[:, -1]  # (batch, num_tokens, hidden_dim)
            
            next_frame_logits = self.to_logits(next_frame_features)  # (batch, num_tokens, vq_num_embeddings)
            
            loss = F.cross_entropy(
                next_frame_logits.reshape(-1, next_frame_logits.size(-1)),
                next_frame_tokens.reshape(-1)
            )
            
            return latent_actions, next_frame_logits, loss
        
        return latent_actions
    
    def encode_actions(self, prev_frames_tokens, next_frame_tokens):
        return self.forward(prev_frames_tokens, next_frame_tokens)[0]
    
    def predict_next_frame(self, prev_frames_tokens, latent_actions=None):
        batch_size = prev_frames_tokens.shape[0]
        history_length = prev_frames_tokens.shape[1]
        num_tokens = prev_frames_tokens.shape[2]
        
        prev_embeds = []
        for i in range(history_length):
            embeds = self.token_embedding(prev_frames_tokens[:, i])
            embeds = embeds + self.pos_embedding
            prev_embeds.append(embeds)
        
        # If no latent actions provided, use zeros
        if latent_actions is None:
            latent_actions = torch.zeros(
                batch_size, history_length, self.hidden_dim, 
                device=prev_frames_tokens.device
            )
        
        decoder_input = []
        for i in range(history_length):
            frame_embed = prev_embeds[i]
            action_embed = latent_actions[:, i].unsqueeze(1).expand(-1, num_tokens, -1)
            combined = frame_embed + action_embed
            decoder_input.append(combined)
        
        decoder_input = torch.stack(decoder_input, dim=1)
        
        decoded = self.decoder(decoder_input)
        next_frame_features = decoded[:, -1]
        
        next_frame_logits = self.to_logits(next_frame_features)
        next_frame_tokens = torch.argmax(next_frame_logits, dim=-1)
        
        return next_frame_tokens

    def generate(self, prev_tokens, actions, num_steps=10, temperature=0.8):
        batch_size = prev_tokens.shape[0]
        num_tokens = prev_tokens.shape[1]
        
        # Start with all masked tokens
        next_tokens = torch.full((batch_size, num_tokens), self.mask_token_id, 
                                device=prev_tokens.device)
        
        # Iteratively decode
        for step in range(num_steps):
            ratio = 1.0 - 0.5 * (1.0 + np.cos(np.pi * step / num_steps))
            num_mask = max(1, int(num_tokens * (1.0 - ratio)))
            
            logits = self.forward(prev_tokens, actions, next_tokens=None)
            
            scaled_logits = logits / temperature
            
            probs = F.softmax(scaled_logits, dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, self.vq_num_embeddings), 1)
            sampled = sampled.reshape(batch_size, num_tokens)
            
            confidence = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
            
            mask = next_tokens == self.mask_token_id
            confidence = confidence * mask
            
            confidence[~mask] = float('inf')
            
            _, indices = torch.topk(-confidence, num_mask, dim=1)
            
            for b in range(batch_size):
                next_tokens[b, indices[b]] = sampled[b, indices[b]]
            
            if (next_tokens == self.mask_token_id).sum() == 0:
                break
                
        return next_tokens

# Dynamics Model

class MaskGIT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.vq_num_embeddings = config.vq_num_embeddings
        
        self.token_embedding = nn.Embedding(config.vq_num_embeddings + 1, config.hidden_dim)  # +1 for mask token
        
        self.action_embedding = nn.Embedding(config.num_actions, config.hidden_dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, config.hidden_dim))  # 8x8=64 tokens
        
        self.transformer = SpatiotemporalTransformer(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers
        )
        
        self.to_logits = nn.Linear(config.hidden_dim, config.vq_num_embeddings)
        
        self.mask_token_id = config.vq_num_embeddings
        
        self.action_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        self.action_pos_mod = nn.Linear(config.hidden_dim, 64)  # 64 tokens
        
        self.context_length = min(config.dynamics_context_frames, config.history_length)
        
        self.frame_pos_embed = nn.Parameter(torch.randn(1, self.context_length, 1, config.hidden_dim))

    def forward(self, prev_tokens, actions, vqvae_model, next_tokens=None, mask_ratio=0.5, prev_frames_history=None, epoch=None):
        batch_size = prev_tokens.shape[0]
        num_tokens = prev_tokens.shape[1]
        
        # Ensure actions are within valid range
        actions = torch.clamp(actions, 0, self.action_embedding.num_embeddings - 1)
        
        if prev_frames_history is not None:
            context_frames = prev_frames_history[:, -self.context_length:].to(prev_tokens.device)
            actual_context_len = context_frames.shape[1]
        else:
            context_frames = prev_tokens.unsqueeze(1)  # (batch, 1, num_tokens)
            actual_context_len = 1
            
        frame_embeds = []
        for i in range(actual_context_len):
            frame_tokens = torch.clamp(context_frames[:, i], 0, self.token_embedding.num_embeddings - 1)
            token_embeds = self.token_embedding(frame_tokens) + self.pos_embedding
            frame_embeds.append(token_embeds)
            
        transformer_input = torch.stack(frame_embeds, dim=1)  # (batch, context_len, num_tokens, hidden_dim)
        
        if transformer_input.size(1) <= self.frame_pos_embed.size(1):
            transformer_input = transformer_input + self.frame_pos_embed[:, :actual_context_len]
        else:
            transformer_input = transformer_input + self.frame_pos_embed.expand(batch_size, -1, num_tokens, -1)[:, :transformer_input.size(1)]
        
        action_embeds = self.action_embedding(actions).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, hidden_dim)
        transformer_input = transformer_input + action_embeds
        
        transformed = self.transformer(transformer_input)  # (batch, context_len, num_tokens, hidden_dim)
        
        transformed = transformed[:, -1]  # (batch, num_tokens, hidden_dim)
        
        logits = self.to_logits(transformed)  # (batch, num_tokens, vq_num_embeddings)
        
        if next_tokens is not None:
            masked_next_tokens = next_tokens.clone()
            
            mask = torch.rand(batch_size, num_tokens, device=next_tokens.device) < mask_ratio
            
            masked_next_tokens[mask] = self.mask_token_id
            
            # Compute loss only on masked positions
            loss = F.cross_entropy(
                logits.reshape(-1, self.vq_num_embeddings)[mask.reshape(-1)], 
                next_tokens.reshape(-1)[mask.reshape(-1)]
            )
            
            consistency_weight = 0.1 * (1.0 - min(1.0, epoch / 10))  # Decrease weight over 10 epochs

            # Add consistency regularization only after some initial training
            if epoch >= 5:
                with torch.no_grad():
                    generated_tokens = self.generate(prev_tokens, actions, vqvae_model, num_steps=5)
                    
                    reconstructed = vqvae_model.decode(generated_tokens)
                    re_encoded_tokens = vqvae_model.encode(reconstructed)
                
                consistency_loss = F.cross_entropy(
                    logits.reshape(-1, self.vq_num_embeddings),
                    re_encoded_tokens.reshape(-1)
                )
                
                loss = loss + consistency_weight * consistency_loss 
            
            return logits, loss
        
        return logits

    def generate(self, prev_tokens, actions, vqvae_model, num_steps=10, temperature=0.8, prev_frames_history=None):
        batch_size = prev_tokens.shape[0]
        num_tokens = prev_tokens.shape[1]
        
        next_tokens = torch.full((batch_size, num_tokens), self.mask_token_id, 
                                device=prev_tokens.device)
        
        # Iteratively decode
        for step in range(num_steps):
            ratio = 1.0 - 0.5 * (1.0 + np.cos(np.pi * step / num_steps))
            num_mask = max(1, int(num_tokens * (1.0 - ratio)))
            
            logits = self.forward(prev_tokens, actions, vqvae_model, next_tokens=None, prev_frames_history=prev_frames_history)
            
            scaled_logits = logits / temperature
            
            probs = F.softmax(scaled_logits, dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, self.vq_num_embeddings), 1)
            sampled = sampled.reshape(batch_size, num_tokens)
            
            confidence = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
            
            mask = next_tokens == self.mask_token_id
            confidence = confidence * mask
            
            confidence[~mask] = float('inf')
            
            _, indices = torch.topk(-confidence, num_mask, dim=1)
            
            for b in range(batch_size):
                next_tokens[b, indices[b]] = sampled[b, indices[b]]
            
            # Break if no more masks
            if (next_tokens == self.mask_token_id).sum() == 0:
                break
                
        return next_tokens

# Data Collection and Processing

class CoinRunDataset(Dataset):
    def __init__(self, data_dir, image_size=64, history_length=16, transform=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.history_length = history_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Get all episode directories
        self.episodes = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))]
        
        # Build index mapping (episode_idx, frame_idx) -> dataset_idx
        self.index_map = []
        for episode_idx, episode_dir in enumerate(self.episodes):
            frame_files = sorted([f for f in os.listdir(episode_dir) if f.endswith('.png')])
            action_file = os.path.join(episode_dir, 'actions.npy')
            
            # Check if we have enough frames and the actions file
            if len(frame_files) >= history_length + 1 and os.path.exists(action_file):
                for i in range(len(frame_files) - history_length):
                    self.index_map.append((episode_idx, i))
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        episode_idx, frame_idx = self.index_map[idx]
        episode_dir = self.episodes[episode_idx]
        
        # Load frames
        frames = []
        for i in range(self.history_length + 1):  # +1 for the next frame
            frame_path = os.path.join(episode_dir, f"frame_{frame_idx + i:05d}.png")
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        
        # Load action
        # In the paper they use the LAM to predict actions, here I train the lam but cheat and use the actions from the dataset for better results, you can toggle this
        actions = np.load(os.path.join(episode_dir, 'actions.npy'))
        action = actions[frame_idx + self.history_length - 1] 
        
        history_frames = torch.stack(frames[:-1])  # (history_length, C, H, W)
        next_frame = frames[-1]  # Last frame is the next frame
        
        return {
            'history_frames': history_frames,  # (history_length, C, H, W)
            'next_frame': next_frame,          # (C, H, W)
            'action': action                   # Integer action ID
        }

def collect_coinrun_data(output_dir, num_episodes=500, max_steps=1000):
    os.makedirs(output_dir, exist_ok=True)
    
    env = gym.make('procgen:procgen-coinrun-v0')
    
    for episode in range(num_episodes):
        episode_dir = os.path.join(output_dir, f'episode_{episode:03d}')
        os.makedirs(episode_dir, exist_ok=True)
        
        obs = env.reset()
        actions = []
        
        for step in range(max_steps):
            # Save observation as PNG
            img = Image.fromarray(obs)
            img.save(os.path.join(episode_dir, f'frame_{step:05d}.png'))
            
            # Take random action
            # TODO, repeat action selected for better representation 
            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
            
            actions.append(action)
            
            if done:
                break
        
        # Save actions
        np.save(os.path.join(episode_dir, 'actions.npy'), np.array(actions))
    
    env.close()


# Training functions
def train_vqvae(model, train_loader, config):
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, eta_min=config.lr_min)
    
    model.train()
    model.to(device)
    
    for epoch in range(config.epochs):
        epoch_loss = 0
        recon_loss_total = 0
        vq_loss_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            frames = torch.cat([batch['history_frames'].reshape(-1, 3, config.image_size, config.image_size), 
                              batch['next_frame'].unsqueeze(1).reshape(-1, 3, config.image_size, config.image_size)], dim=0)
            frames = frames.to(device)
            
            optimizer.zero_grad()
            
            x_recon, vq_loss, _ = model(frames)
            recon_loss = F.mse_loss(x_recon, frames)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            recon_loss_total += recon_loss.item()
            vq_loss_total += vq_loss.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(), 
                'recon_loss': recon_loss.item(), 
                'vq_loss': vq_loss.item()
            })
        
        scheduler.step()

        torch.save(model.state_dict(), "vqvae_model.pt")
        
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {epoch_loss/len(train_loader):.4f}, "
              f"Recon Loss: {recon_loss_total/len(train_loader):.4f}, "
              f"VQ Loss: {vq_loss_total/len(train_loader):.4f}")
        
    return model


def train_lam_dynamics(vqvae_model, lam_model, dynamics_model, train_loader, config):
    
    optimizer = optim.AdamW(
        list(lam_model.parameters()) + list(dynamics_model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    def lr_schedule(epoch):
        if epoch < config.warmup_epochs:
            # Linear warmup
            return epoch / config.warmup_epochs
        else:
            # Cosine decay
            return config.lr_min / config.lr + (1 - config.lr_min / config.lr) * \
                   (1 + np.cos(np.pi * (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs))) / 2
                   
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    vqvae_model.eval()  # VQVAE is frozen during this training
    lam_model.train()
    dynamics_model.train()
    
    # Move models to device
    vqvae_model.to(device)
    lam_model.to(device)
    dynamics_model.to(device)
    
    for epoch in range(config.epochs):
        lam_loss_total = 0
        dynamics_loss_total = 0
        total_loss_total = 0
        total_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, batch in enumerate(progress_bar):

            history_frames = batch['history_frames'].to(device)
            next_frame = batch['next_frame'].to(device)
            ground_truth_actions = batch['action'].to(device)
            
            batch_size = history_frames.shape[0]
            history_length = history_frames.shape[1]
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                history_frame_tokens = []
                for i in range(history_length):
                    tokens = vqvae_model.encode(history_frames[:, i])
                    history_frame_tokens.append(tokens)
                
                history_frame_tokens = torch.stack(history_frame_tokens, dim=1)
                
                next_frame_tokens = vqvae_model.encode(next_frame)

            latent_actions, next_frame_logits, lam_loss = lam_model(
                history_frame_tokens, 
                next_frame_tokens
            )

            # Uncomment this if you want to use the LAM true to the paper
            '''
            _, dynamics_loss = dynamics_model(
                history_frame_tokens[:, -1],  # Last frame as primary input
                latent_actions,
                vqvae_model,
                next_frame_tokens,
                prev_frames_history=history_frame_tokens,  # Pass full history
                epoch=epoch  # Pass epoch for consistency weight scaling
            )
            '''
            
            # Train dynamics model with ground truth actions
            _, dynamics_loss = dynamics_model(
                history_frame_tokens[:, -1],  # Last frame as primary input
                ground_truth_actions,
                vqvae_model,
                next_frame_tokens,
                prev_frames_history=history_frame_tokens,
                epoch=epoch  # Pass epoch for consistency weight scaling
            )
            
            total_loss = lam_loss + dynamics_loss

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(lam_model.parameters()) + list(dynamics_model.parameters()), 
                max_norm=1.0
            )
            
            optimizer.step()
            
            lam_loss_total += lam_loss.item()
            dynamics_loss_total += dynamics_loss.item()
            total_loss_total += total_loss.item()
            total_batches += 1
            
            progress_bar.set_postfix({
                'total_loss': total_loss.item(),
                'lam_loss': lam_loss.item(),
                'dyn_loss': dynamics_loss.item()
            })
            
            # Periodic checkpoint
            if batch_idx % 200 == 0:
                torch.save(dynamics_model.state_dict(), "dynamics_model.pt")
                torch.save(lam_model.state_dict(), "lam_model.pt")
        
        scheduler.step()
        
        # Epoch summary
        print(f"Epoch {epoch+1}/{config.epochs}, "
              f"Total Loss: {total_loss_total/len(train_loader):.4f}, "
              f"LAM Loss: {lam_loss_total/len(train_loader):.4f}, "
              f"Dynamics Loss: {dynamics_loss_total/len(train_loader):.4f}")

        # Always save current model
        torch.save(dynamics_model.state_dict(), f"dynamics_model_epoch_{epoch+1}.pt")
    
    return lam_model, dynamics_model

# Main function
def main():
    # Create data directory
    data_dir = "coinrun_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Collect data if needed
    if len(os.listdir(data_dir)) == 0:
        print("Collecting CoinRun data...")
        collect_coinrun_data(data_dir)
    
    # Create dataset and dataloader
    dataset = CoinRunDataset(data_dir, image_size=config.image_size, history_length=config.history_length)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Initialize models
    vqvae_model = VQVAEModel(config)
    lam_model = LatentActionModel(config)
    dynamics_model = MaskGIT(config)

    if os.path.exists("vqvae_model.pt"):
        print("Loading pre-trained vt model...")
        vqvae_model.load_state_dict(torch.load("vqvae_model.pt"))
    else:
        print("Training VQVAE...")
        vqvae_model = train_vqvae(vqvae_model, train_loader, config)
        torch.save(vqvae_model.state_dict(), "vqvae_model.pt")
    
    # Check if models exist and load them, otherwise train
    if os.path.exists("lam_model.pt") and os.path.exists("dynamics_model.pt"):
        print("Loading pre-trained lam dyn models...")
        lam_model.load_state_dict(torch.load("lam_model.pt"))
        dynamics_model.load_state_dict(torch.load("dynamics_model.pt"))
    else:
        print("Jointly training Latent Action Model and Dynamics Model...")
        lam_model, dynamics_model = train_lam_dynamics(vqvae_model, lam_model, dynamics_model, train_loader, config)
        torch.save(lam_model.state_dict(), "lam_model.pt")
        torch.save(dynamics_model.state_dict(), "dynamics_model.pt")

if __name__ == "__main__":
    main()