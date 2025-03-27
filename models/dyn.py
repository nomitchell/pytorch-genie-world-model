import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.stt import SpatiotemporalTransformer

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
