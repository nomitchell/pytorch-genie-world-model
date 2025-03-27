import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.stt import SpatiotemporalTransformer

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