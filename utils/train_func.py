import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn import utils as nn_utils
from tqdm import tqdm
import numpy as np

def train_vqvae(model, train_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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