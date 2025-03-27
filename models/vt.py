import torch
import torch.nn as nn
import torch.nn.functional as F

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